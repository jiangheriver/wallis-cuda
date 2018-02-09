/*
Wallis (CUDA accelerate)

Author:		Jiang He (GEOVIS)
Email:		riverj AT yeah DOT net
GitHub:		https://github.com/jiangheriver/wallis-cuda
Homepage:	https://jiangheriver.github.io
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//headers for OpenMP, CUDA and GDAL
#include <omp.h>
#include <cuda_runtime_api.h>
#include <gdal_priv.h>

//handle for "BlockProcess" threads and mutex for GDAL RasterIO
//check the complier is ViusalStudio's MSC or Linux's GCC
#ifdef _MSC_VER
#include <Windows.h>
#include <WinBase.h>
DWORD WINAPI BlockThread(PVOID pvParam);
HANDLE g_hMutexRasterIO;
#else
#include <pthread.h>
#define min(a,b) std::min(a,b)
#define max(a,b) std::max(a,b)
void *BlockThread(void *pvParam);
sem_t g_semRasterIO;
#endif

//for GeoTiff or other formats, usually read specifies numbers of image lines as one block,
//this number set to 1024 in my program
#define READ_BLOCK_LINE 1024

//the recommend neighbor window size for wallis is recommendly set to an odd number between
//30 to 50 in related documents of image process, since 32*32=1024, equals to 
//the maximum number of threads per GPU thread block (1024, when CUDA compute capability >= 2.0)
//so I set GRID_LENGTH to 32, means the pixel's neighbor windows size is 31
//for point (X,Y), the rectangle for calculate R0 and R1 is rect(X-15,Y-15,X+15,Y+15)
#define GRID_LENGTH 32

//brightness coeff (B_VALUE), constract coeff (C_VALUE), target sigma (TARGET_SIGMA_VALUE)
//in wallis filter
#define B_VALUE 0.5f
#define C_VALUE 0.8f
#define TARGET_SIGMA_VALUE 60.0f

//this alignment is for store R0,R1 value in CUDA texture memory
#define ALIGNMENT 512

//maximum support raster image's width
#define REQUIRE_SUPPORT_RASTER_WIDTH 40000

//as the source code belows mentioned, for image lines' block ("main" area), we need addtional
//each GRID_LENGTH/2 lines for "pre" and "post" area
const int GRID_HALF_LENGTH = GRID_LENGTH / 2;
const int PROCESS_EXTEND_LENGTH = READ_BLOCK_LINE + GRID_LENGTH;

//R0 and R1 were stored in float2 structure
//calculate texture buffer's width and height for store R0 and R1, include padding for alignment, ceiling...
const int CUDA_GRID_ALIGNMENT = ALIGNMENT / sizeof(float2);
const int CUDA_GRID_SIZE_X = (REQUIRE_SUPPORT_RASTER_WIDTH / GRID_LENGTH / CUDA_GRID_ALIGNMENT)*CUDA_GRID_ALIGNMENT +\
				 CUDA_GRID_ALIGNMENT;
const int CUDA_GRID_SIZE_Y = (PROCESS_EXTEND_LENGTH - 1) / GRID_LENGTH + 1;

//struct use by process threade
typedef struct PROCESS_BLOCK
{
	unsigned char *pucBlockExtendBuffer;	//image lines' block buffer, include "main","pre" and "post"
	int iWidth;	//Image width, include padding bytes for alignment
	int iRasterSizeX;	//Actual image width
	int iPosY;	//Current process image line index
	int iProcessSizeY;	//numbers of image line in this process routine, "main" area only
	int iGridSizeX; //texture buffer's width for store R0 and R1, NOT including padding for alignment
}PROCESS_BLOCK_TAG;

//pointer and image buffer for GDAL image output
GDALRasterBand *g_pBandOutput;
unsigned char *g_pucWriteBuffer;

//cuda_texPixel,cuda_formatDestPixel,cuda_textBlockArrayT: use for texture memory (image pixel buffer, 
//								copy from pucBlockExtendBuffer)
//cuda_texR0R1,cuda_formatDescR0R1,cuda_textR0R1ArrayT: use for texture memory (store R0 and R1)
texture<unsigned char, 2> cuda_texPixel;
texture<float2, 2> cuda_texR0R1;

cudaChannelFormatDesc cuda_formatDescPixel;
cudaChannelFormatDesc cuda_formatDescR0R1;

cudaArray_t cuda_texBlockArrayT;
cudaArray_t cuda_texR0R1ArrayT;

//output block buffer after wallis process, will copy to g_pucWriteBuffer
unsigned char *cuda_pucBlockOutputBuffer;

//buffer for store R0 and R1, not in texture, this buffer is update in cuda_wallis_grid, and will copy to texture
//before cuda_wallis_inner
float2 *cuda_pf2R0R1Buffer;

//constant values use in GPU, the meaning is same as above
__constant__ float cuda_B_VALUE = B_VALUE;
__constant__ float cuda_C_VALUE = C_VALUE;
__constant__ float cuda_TARGET_SIGMA_VALUE = TARGET_SIGMA_VALUE;

__constant__ int cuda_GRID_HALF_LENGTH = GRID_HALF_LENGTH;
__constant__ int cuda_FILTER_WINDOW_LENGTH = GRID_LENGTH - 1;
__constant__ int cuda_GRID_SIZE_X = CUDA_GRID_SIZE_X;
__constant__ int cuda_REDUCTION_BEGIN = GRID_LENGTH*GRID_LENGTH / 2;

//these are some derived constant values, calculate first
__constant__ float cuda_C_MUL_SIGMA_VALUE = C_VALUE*TARGET_SIGMA_VALUE;
__constant__ float cuda_1_C_MUL_SIGMA_VALUE = (1.0f - C_VALUE)*TARGET_SIGMA_VALUE;
__constant__ float cuda_B_MUL_127_5_VALUE = B_VALUE*127.5f;
__constant__ float cuda_1_MINUS_B_VALUE = 1.0f - B_VALUE;
__constant__ float cuda_1_DIV_GRID_LENGTH = 1.0f / ((float)GRID_LENGTH);

//function for calculate R0 and R1
__global__ void cuda_wallis_grid(float2 *pf2R0R1Buffer, int iGridSizeX, int iGridSizeY);

//function for use R0 and R1 to calculate pixels value after wallis filtered
__global__ void cuda_wallis_inner(unsigned char *pucBlockOutputBuffer, int iWidth);

int main(int argc, char *argv[])
{
	//input arguments...
	if ((argc<2) || (argc>3))
	{
		puts("usage: Wallis <inputFile> [<outputFile>]");
		return -1;
	}

	time_t timeBegin = time(NULL);
	omp_set_num_threads(omp_get_num_procs());

	char *pcInputFilename = argv[1];
	std::string strOutputFilename;

	if (argc == 2)
	{
		const char *pcLastDot = strrchr(pcInputFilename, '.');
		if (pcLastDot == NULL)
		{
			puts("Invalid inputFile!");
			return -2;
		}

		strOutputFilename = std::string(pcInputFilename, pcLastDot - pcInputFilename);
		strOutputFilename += "_wallis.tif";
	}
	else
	{
		strOutputFilename = argv[2];
	}

	//GDAL initialize and open source image
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");

	GDALDataset *pDatasetInput = (GDALDataset*)GDALOpen(pcInputFilename, GA_ReadOnly);
	if (pDatasetInput == NULL)
	{
		puts("inputFile open failed!");
		return -3;
	}

	//my program support single band 16bits/pixel format only
	if (pDatasetInput->GetRasterCount() != 1)
	{
		GDALClose(pDatasetInput);
		puts("Support single band only!");
		return -4;
	}

	GDALRasterBand *pBand = pDatasetInput->GetRasterBand(1);
	if (pBand->GetRasterDataType() != GDT_UInt16)
	{
		GDALClose(pDatasetInput);
		puts("Support UInt16 only!");
		return -5;
	}

	//check image's width and height
	int iRasterSizeX = pDatasetInput->GetRasterXSize();
	int iRasterSizeY = pDatasetInput->GetRasterYSize();

	int iMaxSupportWidth = (CUDA_GRID_SIZE_X - 1)*GRID_LENGTH;
	if (iRasterSizeX > iMaxSupportWidth)
	{
		GDALClose(pDatasetInput);
		printf("Support width<=%d only!\n", iMaxSupportWidth);
		return -6;
	}

	//create output image file
	GDALDataset *pDatasetOutput = GetGDALDriverManager()->GetDriverByName("GTIFF")->Create(\
					strOutputFilename.c_str(), iRasterSizeX, iRasterSizeY, \
					1, GDT_Byte, NULL);
	if (pDatasetOutput == NULL)
	{
		GDALClose(pDatasetInput);
		puts("Create wallis file failed!");
		return -7;
	}

	//duplicate geo and projection info
	double f64GeoTransformArray[6];
	pDatasetInput->GetGeoTransform(f64GeoTransformArray);
	pDatasetOutput->SetGeoTransform(f64GeoTransformArray);

	const char *pcProjectionRef = pDatasetInput->GetProjectionRef();
	if ((pcProjectionRef != NULL) && (strlen(pcProjectionRef)>0))
	{
		pDatasetOutput->SetProjection(pcProjectionRef);
	}

	//alloc cuda_pucBlockOutputBuffer, and get the width after padding for alignment (nWidth)
	size_t nWidth;
	if (cudaMallocPitch((void**)&cuda_pucBlockOutputBuffer, &nWidth, (size_t)iRasterSizeX, READ_BLOCK_LINE) \
		!= cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc global memory failed!");
		cudaDeviceReset();
		return -8;
	}

	//alloc cuda_pf2R0R1Buffer
	if (cudaMalloc((void**)&cuda_pf2R0R1Buffer, CUDA_GRID_SIZE_X*CUDA_GRID_SIZE_Y*sizeof(float2)) != cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc global memory failed!");
		cudaDeviceReset();
		return -8;
	}

	//calc total grids in x-axis
	g_pBandOutput = pDatasetOutput->GetRasterBand(1);
	int iGridSizeX = (iRasterSizeX + (GRID_LENGTH - 1)) / GRID_LENGTH + 1;

	//compute normalized (range [0,255]), coeff from source image, and set GDAL cache
	double f64MinValue;
	double f64MaxValue;
	double f64MeanValue;
	double f64StdDev;

	GDALSetCacheMax64((GIntBig)nWidth* (iRasterSizeY + GRID_LENGTH) * 4);
	pBand->ComputeStatistics(TRUE, &f64MinValue, &f64MaxValue, &f64MeanValue, &f64StdDev, NULL, NULL);

	float f32MinValue = (float)f64MinValue;
	float f32NormalizeCoeff = 255 / ((float)f64MaxValue - f32MinValue);

	if (f64MaxValue - f64MinValue <= 1)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("Invalid min&max pixel values!");
		cudaDeviceReset();
		return -9;
	}

	int iWidth = (int)nWidth;
	int iPadSize = iWidth - iRasterSizeX;

	//texture memory for store image lines' block, the format is 8bit unsigned char,
	//set filterMode (Point) and addressMode (Clamp)
	cuda_texPixel.normalized = 0;
	cuda_texPixel.filterMode = cudaFilterModePoint;
	cuda_texPixel.addressMode[0] = cudaAddressModeClamp;
	cuda_texPixel.addressMode[1] = cudaAddressModeClamp;

	cuda_formatDescPixel = cudaCreateChannelDesc<unsigned char>();
	if (cudaMallocArray(&cuda_texBlockArrayT, &cuda_formatDescPixel, nWidth, (size_t)PROCESS_EXTEND_LENGTH)\
		 != cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc texture memory failed!");
		cudaDeviceReset();
		return -10;
	}

	//texture memory for store R0 and R1, the format is float2 stucture,
	//set filterMode (Linear) and addressMode (Clamp)
	cuda_texR0R1.normalized = 0;
	cuda_texR0R1.filterMode = cudaFilterModeLinear;
	cuda_texR0R1.addressMode[0] = cudaAddressModeClamp;
	cuda_texR0R1.addressMode[1] = cudaAddressModeClamp;

	cuda_formatDescR0R1 = cudaCreateChannelDesc<float2>();
	if (cudaMallocArray(&cuda_texR0R1ArrayT, &cuda_formatDescR0R1, CUDA_GRID_SIZE_X, CUDA_GRID_SIZE_Y)\
		 != cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc texture memory failed!");
		cudaDeviceReset();
		return -10;
	}

	//set read (input) and write (output) buffers
	size_t nInSize = iRasterSizeX*READ_BLOCK_LINE;
	size_t nOutSize = nWidth*READ_BLOCK_LINE;
	size_t nExtendSize = nWidth*PROCESS_EXTEND_LENGTH;

	g_pucWriteBuffer = new unsigned char[nOutSize];
	unsigned short *pusReadBuffer = new unsigned short[nInSize];

	//"ping pong" mode, when reading BlockB, processing BlockA at same time;
	//when reading BlockA, processing BlockB at same time...
	bool bBlockTag = true;
	PROCESS_BLOCK processBlockA;
	PROCESS_BLOCK processBlockB;

	processBlockA.iWidth = iWidth;
	processBlockB.iWidth = iWidth;

	processBlockA.iRasterSizeX = iRasterSizeX;
	processBlockB.iRasterSizeX = iRasterSizeX;

	processBlockA.iGridSizeX = iGridSizeX;
	processBlockB.iGridSizeX = iGridSizeX;

	processBlockA.pucBlockExtendBuffer = new unsigned char[nExtendSize];
	processBlockB.pucBlockExtendBuffer = new unsigned char[nExtendSize];

	//create 3 reader buffers , "pre->main->post->pre->main->post->..." ring
	unsigned char *pucReadBufferArray[3];
	for (int i = 0; i<3; i++)
	{
		pucReadBufferArray[i] = new unsigned char[nOutSize];
	}

	if ((processBlockA.pucBlockExtendBuffer == NULL) || (processBlockB.pucBlockExtendBuffer == NULL) || \
		(g_pucWriteBuffer == NULL) || (pusReadBuffer == NULL) || (pucReadBufferArray[0] == NULL) || \
		(pucReadBufferArray[1] == NULL) || (pucReadBufferArray[2] == NULL))
	{
		if (processBlockA.pucBlockExtendBuffer != NULL)
		{
			delete[] processBlockA.pucBlockExtendBuffer;
		}

		if (processBlockB.pucBlockExtendBuffer != NULL)
		{
			delete[] processBlockB.pucBlockExtendBuffer;
		}

		if (g_pucWriteBuffer != NULL)
		{
			delete[] g_pucWriteBuffer;
		}

		if (pusReadBuffer != NULL)
		{
			delete[] pusReadBuffer;
		}

		for (int i = 0; i<3; i++)
		{
			if (pucReadBufferArray[i] != NULL)
			{
				delete[] pucReadBufferArray[i];
			}
		}

		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc memory failed!");
		cudaDeviceReset();
		return -11;
	}

        //the bottom GRID_HALF_LENGTH lines of "pre" buffer + READ_BLOCK_LINE lines of "main" buffer+the top READ_BLOCK_LINE lines of "post" buffer
	//  ==> Extend Buffer
        size_t nPrePostSize = nWidth*GRID_HALF_LENGTH;
        size_t nMainSize = nWidth*READ_BLOCK_LINE;
        size_t nPreOffset = nWidth*(READ_BLOCK_LINE - GRID_HALF_LENGTH);
	size_t nPreAndMainSize = nPrePostSize + nMainSize;

	//create handle for process thread and mutex 
#ifdef _MSC_VER
	HANDLE hThreadLast = NULL;
	g_hMutexRasterIO = CreateMutex(NULL, FALSE, NULL);
#else
	pthread_t hThreadLast = NULL;
	sem_init(&g_semRasterIO, 0, 0);
#endif

	//main process loop
	for (int iLine = 0; iLine<iRasterSizeY; iLine += READ_BLOCK_LINE)
	{
		//calculate lines need to read, check if it is bottom block or not
		int iReadLine = iRasterSizeY - iLine;
		bool bMostBottomLines = true;

		if (iReadLine>READ_BLOCK_LINE)
		{
			iReadLine = READ_BLOCK_LINE;
			bMostBottomLines = false;
		}

		size_t nSize=nWidth*iReadLine;
                int iIndexPost = iLine / READ_BLOCK_LINE;

		//the block being read would be the "post" buffer
		//block before it, also means block have just read, would be the "main" buffer
		//block before "main" buffer would be the "pre" buffer
		unsigned char *pucPostBuffer = pucReadBufferArray[iIndexPost % 3];
		unsigned char *pucMainBuffer = pucReadBufferArray[(iIndexPost + 2) % 3];
		unsigned char *pucPreBuffer = pucReadBufferArray[(iIndexPost + 1) % 3];

		//read image lines from file
#ifdef _MSC_VER
		WaitForSingleObject(g_hMutexRasterIO, INFINITE);
		pBand->RasterIO(GF_Read, 0, iLine, iRasterSizeX, iReadLine, pusReadBuffer, \
				iRasterSizeX, iReadLine, GDT_UInt16, 0, 0);
		ReleaseMutex(g_hMutexRasterIO);
#else
		sem_wait(&g_semRasterIO);
		pBand->RasterIO(GF_Read, 0, iLine, iRasterSizeX, iReadLine, pusReadBuffer, \
				iRasterSizeX, iReadLine, GDT_UInt16, 0, 0);
		sem_post(&g_semRasterIO);
#endif

		//normalize pixels' value to range 0-255 (byte)
#pragma omp parallel for
		for (int i = 0; i < iReadLine; i++)
		{
			unsigned short *pusInput = pusReadBuffer + (i*iRasterSizeX);
			unsigned char *pucOutput = pucPostBuffer + (i*iWidth);

			for (int j = 0; j < iRasterSizeX; j++)
			{
				*pucOutput = (unsigned char)max(min(f32NormalizeCoeff* \
						((float)(*pusInput) - f32MinValue) + 0.5f, 255.0f), 0.0f);
				pusInput++;
				pucOutput++;
			}

			memset(pucOutput, 0, iPadSize);
		}

		//the "main" buffer is valid when iIndexPost>0
		if (iIndexPost>0)
		{
			//ping pong mode
			PROCESS_BLOCK *pBlockCurrent = bBlockTag ? (&processBlockA) : (&processBlockB);
			bBlockTag = bBlockTag ? false : true;

			//iLine is the begin image line of "post",
			// so the process image line index (of "main") is iLine-READ_BLOCK_LINE
			unsigned char *pucExtend = pBlockCurrent->pucBlockExtendBuffer;
			pBlockCurrent->iPosY = iLine - READ_BLOCK_LINE;
			pBlockCurrent->iProcessSizeY = READ_BLOCK_LINE;

			//the bottom GRID_HALF_LENGTH lines of "pre" buffer,
			//if not exists (iIndexPost==1), fill with zero
			if (iIndexPost>1)
			{
				memcpy(pucExtend, pucPreBuffer + nPreOffset, nPrePostSize);
			}
			else
			{
				memset(pucExtend, 0, nPrePostSize);
			}

			//READ_BLOCK_LINE lines of "main" buffer
			memcpy(pucExtend + nPrePostSize, pucMainBuffer, nMainSize);
			pucExtend += nPreAndMainSize;

			//the top READ_BLOCK_LINE lines of "post" buffer
			if (iReadLine < GRID_HALF_LENGTH)
			{
				memcpy(pucExtend, pucPostBuffer, nReadSize);
				memset(pucExtend + nReadSize, 0, nPrePostSize - nReadSize);
			}
			else
			{
				memcpy(pucExtend, pucPostBuffer, nPrePostSize);
			}

			//wait last process thread finished (wallis filter and write to file),
			//then create new one
#ifdef _MSC_VER
			if (hThreadLast)
			{
				WaitForSingleObject(hThreadLast, INFINITE);
				CloseHandle(hThreadLast);
			}

			hThreadLast = CreateThread(NULL, 0, BlockThread, pBlockCurrent, 0, NULL);
#else
			if (hThreadLast)
			{
				pthread_join(hThreadLast, NULL);
			}

			pthread_create(&hThreadLast, NULL, BlockThread, pBlockCurrent);
#endif
		}

		//process the bottom block, this block would copy to "main" buffer,
		//the last "main" buffer would be "pre" now (use its bottom GRID_HALF_LENGTH lines)
		//and fill "post" buffer with zero. other method is similar to above
		if (bMostBottomLines)
		{
			PROCESS_BLOCK *pBlockCurrent = bBlockTag ? (&processBlockA) : (&processBlockB);
			bBlockTag = bBlockTag ? false : true;

			unsigned char *pucExtend = pBlockCurrent->pucBlockExtendBuffer;
			pBlockCurrent->iPosY = iLine;
			pBlockCurrent->iProcessSizeY = iReadLine;

			memcpy(pucExtend, pucMainBuffer + nPreOffset, nPrePostSize);
			pucExtend += nPrePostSize;

			//copy to "main" buffer,
			//maybe it was less than READ_BLOCK_LINE, fill the remain and "post" with zero
			if (iReadLine < READ_BLOCK_LINE)
			{
				memcpy(pucExtend, pucPostBuffer, nReadSize);
				memset(pucExtend + nReadSize, 0, nMainSize + nPrePostSize - nReadSize);
			}
			else
			{
				memcpy(pucExtend, pucPostBuffer, nMainSize);
				memset(pucExtend + nMainSize, 0, nPrePostSize);
			}

#ifdef _MSC_VER
			if (hThreadLast)
			{
				WaitForSingleObject(hThreadLast, INFINITE);
				CloseHandle(hThreadLast);
			}

			hThreadLast = CreateThread(NULL, 0, BlockThread, pBlockCurrent, 0, NULL);
#else
			if (hThreadLast)
			{
				pthread_join(hThreadLast, NULL);
			}

			pthread_create(&hThreadLast, NULL, BlockThread, pBlockCurrent);
#endif
		}
	}

	//wait the last process thread to finish	
	if (hThreadLast)
	{
#ifdef _MSC_VER
		WaitForSingleObject(hThreadLast, INFINITE);
		CloseHandle(hThreadLast);
		CloseHandle(g_hMutexRasterIO);
#else
		pthread_join(hThreadLast, NULL);
		sem_destroy(g_hMutexRasterIO);
#endif
	}

	//release buffers, close handles, etc...
	delete[] g_pucWriteBuffer;
	delete[] pusReadBuffer;

	delete[] processBlockA.pucBlockExtendBuffer;
	delete[] processBlockB.pucBlockExtendBuffer;

	for (int i = 0; i<3; i++)
	{
		delete[] pucReadBufferArray[i];
	}

	GDALClose(pDatasetInput);
	GDALClose(pDatasetOutput);

	printf("cost %d seconds...\n", (int)(time(NULL) - timeBegin));
	cudaDeviceReset();
	return 0;
}

	//process thread
#ifdef _MSC_VER
DWORD WINAPI BlockThread(PVOID pvParam)
#else
void *BlockThread(void *pvParam)
#endif
{
	PROCESS_BLOCK *pBlockCurrent = (PROCESS_BLOCK*)pvParam;
	int iWidth = pBlockCurrent->iWidth;
	int iRasterSizeX = pBlockCurrent->iRasterSizeX;
	int iPosY = pBlockCurrent->iPosY;
	int iProcessSizeY = pBlockCurrent->iProcessSizeY;
	int iGridSizeX = pBlockCurrent->iGridSizeX;
	int iGridSizeY = (iProcessSizeY + (GRID_LENGTH - 1)) / GRID_LENGTH + 1;

	//example: if the image is 32000*32000, GRID_LENGTH=32,
	//	then the CUDA thread block to calculate R0 and R1 would be 1001*1001;
	//	but when use R0 and R1 to calulate wallis filtered pixels, 
	//	the CUDA thread block would be 1000*1000;
	//	CUDA threads in CUDA thread block is 32*32
	dim3 dimWallisGridBlocks(iGridSizeX, iGridSizeY);
	dim3 dimWallisInnerBlocks(iGridSizeX - 1, iGridSizeY - 1);
	dim3 dimThreads(GRID_LENGTH, GRID_LENGTH);

	//copy buffer to texture, calculate R0 and R1 in GPU
	//y-axis would extend GRID_HALF_LENGTH lines ("pre" and "post"), x-axis not extend
	cudaMemcpyToArray(cuda_texBlockArrayT, 0, 0, pBlockCurrent->pucBlockExtendBuffer, \
				iWidth*(iProcessSizeY + GRID_LENGTH), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(cuda_texPixel, cuda_texBlockArrayT, cuda_formatDescPixel);
	cuda_wallis_grid <<<dimWallisGridBlocks, dimThreads >>>(cuda_pf2R0R1Buffer, iGridSizeX, iGridSizeY);

	//copy R0 and R1 to texture, calculate wallis filtered value in GPU
	cudaMemcpyToArray(cuda_texR0R1ArrayT, 0, 0, cuda_pf2R0R1Buffer, 
				sizeof(float2)*CUDA_GRID_SIZE_X*iGridSizeY, cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(cuda_texR0R1, cuda_texR0R1ArrayT, cuda_formatDescR0R1);
	cuda_wallis_inner <<<dimWallisInnerBlocks, dimThreads >>>(cuda_pucBlockOutputBuffer, iWidth);

	//unbind textures and copy wallis filtered pixels' value to system memory to output
	cudaUnbindTexture(cuda_texPixel);
	cudaUnbindTexture(cuda_texR0R1);
	cudaMemcpy(g_pucWriteBuffer, cuda_pucBlockOutputBuffer, iWidth*iProcessSizeY, \
			cudaMemcpyDeviceToHost);

	//write output image line to file
#ifdef _MSC_VER
	WaitForSingleObject(g_hMutexRasterIO, INFINITE);
	g_pBandOutput->RasterIO(GF_Write, 0, iPosY, iRasterSizeX, iProcessSizeY, g_pucWriteBuffer, \
				iRasterSizeX, iProcessSizeY, GDT_Byte, 0, iWidth);
	ReleaseMutex(g_hMutexRasterIO);
#else
	sem_wait(&g_semRasterIO);
	g_pBandOutput->RasterIO(GF_Write, 0, iPosY, iRasterSizeX, iProcessSizeY, g_pucWriteBuffer, \
				iRasterSizeX, iProcessSizeY, GDT_Byte, 0, iWidth);
	sem_post(&g_semRasterIO);
#endif

	printf("Write %d+%d lines...\n", iPosY, iProcessSizeY);
	return NULL;
}

__global__ void cuda_wallis_grid(float2 *pf2R0R1Buffer, int iGridSizeX, int iGridSizeY)
{
	__shared__ int iSumValidArray[GRID_LENGTH*GRID_LENGTH];
	__shared__ int iSumValueArray[GRID_LENGTH*GRID_LENGTH];
	__shared__ int iSumSquareArray[GRID_LENGTH*GRID_LENGTH];

	int iXInGrid = threadIdx.x;
	int iYInGrid = threadIdx.y;
	int iGridOffset = iXInGrid*GRID_LENGTH + iYInGrid;

	int iGridX = blockIdx.x;
	int iGridY = blockIdx.y;

	//for point (iGridX*32,iGridY*32),
	//the rectangle for calculate R0 and R1 is rect(iGridX*32-15,iGridY*32-15,iGridX*32+15,iGridY*32+15)
	//but the rectangle in the CUDA thread block is rect(iGridX*32-16,iGridY*32-16,iGridX*32+15,iGridY*32+15)
	//the most left pixels (iXInGrid==0) and the most top pixels (iYInGrid==0)
	//would not be calculated for rectangle's mean and variance value
	//so I set these pixel's iPixelValue to zero	
	int iPointX = iXInGrid - cuda_GRID_HALF_LENGTH + iGridX*GRID_LENGTH;
	int iPointY = iYInGrid + iGridY*GRID_LENGTH;
	int iPixelValue = ((int)(iXInGrid&&iYInGrid))* ((int)tex2D(cuda_texPixel, iPointX, iPointY));

	//discard pixels which value is zero (means pad pixel in the four corner of raster image,
	//or the most left/top pixels in CUDA thread block)
	iSumValidArray[iGridOffset] = (int)(iPixelValue > 0);
	iSumValueArray[iGridOffset] = iPixelValue;
	iSumSquareArray[iGridOffset] = iPixelValue*iPixelValue;
	__syncthreads();

	//merge operation	
	for (int k = cuda_REDUCTION_BEGIN; k >= 1; k /= 2)
	{
		if (iGridOffset < k)
		{
			int iGridOffsetK = iGridOffset + k;
			iSumValidArray[iGridOffset] += iSumValidArray[iGridOffsetK];
			iSumValueArray[iGridOffset] += iSumValueArray[iGridOffsetK];
			iSumSquareArray[iGridOffset] += iSumSquareArray[iGridOffsetK];
		}

		__syncthreads();
	}

	if (iGridOffset == 0)
	{
		float2 *pf2R0R1 = pf2R0R1Buffer + (iGridY*cuda_GRID_SIZE_X + iGridX);
		pf2R0R1->x = 0.0f;
		pf2R0R1->y = 0.0f;

		int iValidPixelCount = iSumValidArray[0];
		if (iValidPixelCount)
		{
			//calculate mean and variance value
			int iSumSquare = iSumSquareArray[0];
			float f32FilterMean = ((float)iSumValueArray[0]) / iValidPixelCount;
			float f32FilterVariance = (float)(iSumSquare / iValidPixelCount) + \
					((float)(iSumSquare%iValidPixelCount)) / iValidPixelCount - \
					f32FilterMean*f32FilterMean;
			f32FilterVariance *= (float)(f32FilterVariance > 0.0f);

			//calculate R0 and R1 value
			float f32R1 = cuda_C_MUL_SIGMA_VALUE / (cuda_C_VALUE*sqrtf(f32FilterVariance) + \
					cuda_1_C_MUL_SIGMA_VALUE);
			pf2R0R1->x = cuda_B_MUL_127_5_VALUE + (cuda_1_MINUS_B_VALUE - f32R1)*f32FilterMean;
			pf2R0R1->y = f32R1;
		}
	}
}

__global__ void cuda_wallis_inner(unsigned char *pucBlockOutputBuffer, int iWidth)
{
	int iPointX = threadIdx.x + blockIdx.x*blockDim.x;
	int iPointY = threadIdx.y + blockIdx.y*blockDim.y;
	float f32GridX = ((float)iPointX)*cuda_1_DIV_GRID_LENGTH;
	float f32GridY = ((float)iPointY)*cuda_1_DIV_GRID_LENGTH;

	//pixels store in texture memory were begin with "pre" area (GRID_HALF_LENGTH lines),
	//but R0 and R1 were not.
	//get interpolated R0 and R1
	float f32PixelValue = (float)tex2D(cuda_texPixel, iPointX, iPointY + cuda_GRID_HALF_LENGTH);
	float2 f2R0R1 = tex2D(cuda_texR0R1, f32GridX, f32GridY);

	//calculate wallis filtered pixel value, and clamp it to range [0,255]
	int iOutPixelValue = (int)((f2R0R1.y)*f32PixelValue + (f2R0R1.x) + 0.5f);
	iOutPixelValue -= ((int)(iOutPixelValue > 255)*(iOutPixelValue - 255));
	iOutPixelValue *= ((int)((iOutPixelValue > 0) && (f32PixelValue > 0.0f)));
	*(pucBlockOutputBuffer + (iPointY*iWidth + iPointX)) = (unsigned char)iOutPixelValue;
}
