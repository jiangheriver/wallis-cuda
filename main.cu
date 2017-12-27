/*
Wallis (CUDA accelerate)

Author:		Jiang He
Email:		riverj AT yeah DOT net
GitHub:		https://github.com/jiangheriver/wallis-cuda
Homepage:	https://jiangheriver.github.io
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>
#include <cuda_runtime_api.h>
#include <gdal_priv.h>

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

#define READ_BLOCK_LINE 1024
#define GRID_LENGTH 32

#define B_VALUE 0.5f
#define C_VALUE 0.8f
#define TARGET_SIGMA_VALUE 60.0f

#define ALIGNMENT 512
#define REQUIRE_SUPPORT_RASTER_WIDTH 40000

const int GRID_HALF_LENGTH = GRID_LENGTH / 2;
const int PROCESS_EXTEND_LENGTH = READ_BLOCK_LINE + GRID_LENGTH;
const int CUDA_GRID_ALIGNMENT = ALIGNMENT / sizeof(float2);
const int CUDA_GRID_SIZE_X = (REQUIRE_SUPPORT_RASTER_WIDTH / GRID_LENGTH / CUDA_GRID_ALIGNMENT)*CUDA_GRID_ALIGNMENT + CUDA_GRID_ALIGNMENT;
const int CUDA_GRID_SIZE_Y = (PROCESS_EXTEND_LENGTH - 1) / GRID_LENGTH + 1;

typedef struct PROCESS_BLOCK
{
	unsigned char *pucBlockExtendBuffer;
	int iWidth;
	int iRasterSizeX;

	int iPosY;
	int iProcessSizeY;

	int iGridSizeX;
	int iGridSizeY;
}PROCESS_BLOCK_TAG;

GDALRasterBand *g_pBandOutput;
unsigned char *g_pucWriteBuffer;

texture<unsigned char, 2> cuda_texPixel;
texture<float2, 2> cuda_texR0R1;

cudaChannelFormatDesc cuda_formatDescPixel;
cudaChannelFormatDesc cuda_formatDescR0R1;

cudaArray_t cuda_texBlockArrayT;
cudaArray_t cuda_texR0R1ArrayT;

unsigned char *cuda_pucBlockOutputBuffer;
float2 *cuda_pf2R0R1Buffer;

__constant__ float cuda_B_VALUE = B_VALUE;
__constant__ float cuda_C_VALUE = C_VALUE;
__constant__ float cuda_TARGET_SIGMA_VALUE = TARGET_SIGMA_VALUE;

__constant__ int cuda_GRID_HALF_LENGTH = GRID_HALF_LENGTH;
__constant__ int cuda_GRID_LEFT_OFFSET = 1 - GRID_HALF_LENGTH;
__constant__ int cuda_FILTER_WINDOW_LENGTH = GRID_LENGTH - 1;
__constant__ int cuda_GRID_SIZE_X = CUDA_GRID_SIZE_X;
__constant__ int cuda_REDUCTION_BEGIN = GRID_LENGTH*GRID_LENGTH / 2;

__constant__ float cuda_C_MUL_SIGMA_VALUE = C_VALUE*TARGET_SIGMA_VALUE;
__constant__ float cuda_1_C_MUL_SIGMA_VALUE = (1.0f - C_VALUE)*TARGET_SIGMA_VALUE;
__constant__ float cuda_B_MUL_127_5_VALUE = B_VALUE*127.5f;
__constant__ float cuda_1_MINUS_B_VALUE = 1.0f - B_VALUE;
__constant__ float cuda_1_DIV_GRID_LENGTH = 1.0f / ((float)GRID_LENGTH);

__global__ void cuda_wallis_grid(float2 *pf2R0R1Buffer, int iGridSizeX, int iGridSizeY);
__global__ void cuda_wallis_inner(unsigned char *pucBlockOutputBuffer, int iWidth);

int main(int argc, char *argv[])
{
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

	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");

	GDALDataset *pDatasetInput = (GDALDataset*)GDALOpen(pcInputFilename, GA_ReadOnly);
	if (pDatasetInput == NULL)
	{
		puts("inputFile open failed!");
		return -3;
	}

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

	int iRasterSizeX = pDatasetInput->GetRasterXSize();
	int iRasterSizeY = pDatasetInput->GetRasterYSize();
	size_t nWidth;

	int iMaxSupportWidth = (CUDA_GRID_SIZE_X - 1)*GRID_LENGTH;
	if (iRasterSizeX > iMaxSupportWidth)
	{
		GDALClose(pDatasetInput);
		printf("Support width<=%d only!\n", iMaxSupportWidth);
		return -6;
	}

	GDALDataset *pDatasetOutput = GetGDALDriverManager()->GetDriverByName("GTIFF")->Create(strOutputFilename.c_str(), iRasterSizeX, iRasterSizeY, 1, GDT_Byte, NULL);
	if (pDatasetOutput == NULL)
	{
		GDALClose(pDatasetInput);
		puts("Create wallis file failed!");
		return -7;
	}

	double f64GeoTransformArray[6];
	pDatasetInput->GetGeoTransform(f64GeoTransformArray);
	pDatasetOutput->SetGeoTransform(f64GeoTransformArray);

	const char *pcProjectionRef = pDatasetInput->GetProjectionRef();
	if ((pcProjectionRef != NULL) && (strlen(pcProjectionRef)>0))
	{
		pDatasetOutput->SetProjection(pcProjectionRef);
	}

	if (cudaMallocPitch((void**)&cuda_pucBlockOutputBuffer, &nWidth, (size_t)iRasterSizeX, READ_BLOCK_LINE) != cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc global memory failed!");
		cudaDeviceReset();
		return -8;
	}

	if (cudaMalloc((void**)&cuda_pf2R0R1Buffer, CUDA_GRID_SIZE_X*CUDA_GRID_SIZE_Y*sizeof(float2)) != cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc global memory failed!");
		cudaDeviceReset();
		return -8;
	}

	g_pBandOutput = pDatasetOutput->GetRasterBand(1);
	int iGridSizeX = (iRasterSizeX + (GRID_LENGTH - 1)) / GRID_LENGTH + 1;

	double f64MinValue;
	double f64MaxValue;
	double f64MeanValue;
	double f64StdDev;

	GDALSetCacheMax64((GIntBig)nWidth*iGridSizeX*GRID_LENGTH * 4);
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

	cuda_texPixel.normalized = 0;
	cuda_texPixel.filterMode = cudaFilterModePoint;
	cuda_texPixel.addressMode[0] = cudaAddressModeClamp;
	cuda_texPixel.addressMode[1] = cudaAddressModeClamp;

	cuda_formatDescPixel = cudaCreateChannelDesc<unsigned char>();
	if (cudaMallocArray(&cuda_texBlockArrayT, &cuda_formatDescPixel, nWidth, (size_t)PROCESS_EXTEND_LENGTH) != cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc texture memory failed!");
		cudaDeviceReset();
		return -10;
	}

	cuda_texR0R1.normalized = 0;
	cuda_texR0R1.filterMode = cudaFilterModeLinear;
	cuda_texR0R1.addressMode[0] = cudaAddressModeClamp;
	cuda_texR0R1.addressMode[1] = cudaAddressModeClamp;

	cuda_formatDescR0R1 = cudaCreateChannelDesc<float2>();
	if (cudaMallocArray(&cuda_texR0R1ArrayT, &cuda_formatDescR0R1, CUDA_GRID_SIZE_X, CUDA_GRID_SIZE_Y) != cudaSuccess)
	{
		GDALClose(pDatasetInput);
		GDALClose(pDatasetOutput);

		puts("alloc texture memory failed!");
		cudaDeviceReset();
		return -10;
	}

	size_t nInSize = iRasterSizeX*READ_BLOCK_LINE;
	size_t nOutSize = nWidth*READ_BLOCK_LINE;
	size_t nExtendSize = nWidth*PROCESS_EXTEND_LENGTH;

	g_pucWriteBuffer = new unsigned char[nOutSize];
	unsigned short *pusReadBuffer = new unsigned short[nInSize];

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

	unsigned char *pucReadBufferArray[3];
	for (int i = 0; i<3; i++)
	{
		pucReadBufferArray[i] = new unsigned char[nOutSize];
	}

	if ((processBlockA.pucBlockExtendBuffer == NULL) || (processBlockB.pucBlockExtendBuffer == NULL) || (g_pucWriteBuffer == NULL) || (pusReadBuffer == NULL) || (pucReadBufferArray[0] == NULL) || (pucReadBufferArray[1] == NULL) || (pucReadBufferArray[2] == NULL))
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

#ifdef _MSC_VER
	HANDLE hThreadLast = NULL;
	g_hMutexRasterIO = CreateMutex(NULL, FALSE, NULL);
#else
	pthread_t hThreadLast = NULL;
	sem_init(&g_semRasterIO, 0, 0);
#endif

	for (int iLine = 0; iLine<iRasterSizeY; iLine += READ_BLOCK_LINE)
	{
		int iReadLine = iRasterSizeY - iLine;
		bool bMostBottomLines = true;

		if (iReadLine>READ_BLOCK_LINE)
		{
			iReadLine = READ_BLOCK_LINE;
			bMostBottomLines = false;
		}

		int iIndexPost = iLine / READ_BLOCK_LINE;
		unsigned char *pucPostBuffer = pucReadBufferArray[iIndexPost % 3];
		unsigned char *pucMainBuffer = pucReadBufferArray[(iIndexPost + 2) % 3];
		unsigned char *pucPreBuffer = pucReadBufferArray[(iIndexPost + 1) % 3];

#ifdef _MSC_VER
		WaitForSingleObject(g_hMutexRasterIO, INFINITE);
		pBand->RasterIO(GF_Read, 0, iLine, iRasterSizeX, iReadLine, pusReadBuffer, iRasterSizeX, iReadLine, GDT_UInt16, 0, 0);
		ReleaseMutex(g_hMutexRasterIO);
#else
		sem_wait(&g_semRasterIO);
		pBand->RasterIO(GF_Read, 0, iLine, iRasterSizeX, iReadLine, pusReadBuffer, iRasterSizeX, iReadLine, GDT_UInt16, 0, 0);
		sem_post(&g_semRasterIO);
#endif

#pragma omp parallel for
		for (int i = 0; i < iReadLine; i++)
		{
			unsigned short *pusInput = pusReadBuffer + (i*iRasterSizeX);
			unsigned char *pucOutput = pucPostBuffer + (i*iWidth);

			for (int j = 0; j < iRasterSizeX; j++)
			{
				*pucOutput = (unsigned char)max(min(f32NormalizeCoeff*((float)(*pusInput) - f32MinValue) + 0.5f, 255.0f), 0.0f);
				pusInput++;
				pucOutput++;
			}

			memset(pucOutput, 0, iPadSize);
		}

		size_t nReadSize = nWidth*iReadLine;
		size_t nPrePostSize = nWidth*GRID_HALF_LENGTH;
		size_t nMainSize = nWidth*READ_BLOCK_LINE;
		size_t nPreOffset = nWidth*(READ_BLOCK_LINE - GRID_HALF_LENGTH);

		if (iIndexPost>0)
		{
			PROCESS_BLOCK *pBlockCurrent = bBlockTag ? (&processBlockA) : (&processBlockB);
			bBlockTag = bBlockTag ? false : true;

			unsigned char *pucExtend = pBlockCurrent->pucBlockExtendBuffer;
			pBlockCurrent->iPosY = iLine - READ_BLOCK_LINE;
			pBlockCurrent->iProcessSizeY = READ_BLOCK_LINE;

			memcpy(pucExtend, pucPreBuffer + nPreOffset, nPrePostSize);
			pucExtend += nPrePostSize;

			memcpy(pucExtend, pucMainBuffer, nMainSize);
			pucExtend += nMainSize;

			if (iReadLine < GRID_HALF_LENGTH)
			{
				memcpy(pucExtend, pucPostBuffer, nReadSize);
				memset(pucExtend + nReadSize, 0, nPrePostSize - nReadSize);
			}
			else
			{
				memcpy(pucExtend, pucPostBuffer, nPrePostSize);
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

		if (bMostBottomLines)
		{
			PROCESS_BLOCK *pBlockCurrent = bBlockTag ? (&processBlockA) : (&processBlockB);
			bBlockTag = bBlockTag ? false : true;

			unsigned char *pucExtend = pBlockCurrent->pucBlockExtendBuffer;
			pBlockCurrent->iPosY = iLine;
			pBlockCurrent->iProcessSizeY = iReadLine;

			memcpy(pucExtend, pucMainBuffer + nPreOffset, nPrePostSize);
			pucExtend += nPrePostSize;

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

	dim3 dimWallisGridBlocks(iGridSizeX, iGridSizeY);
	dim3 dimWallisInnerBlocks(iGridSizeX - 1, iGridSizeY - 1);
	dim3 dimThreads(GRID_LENGTH, GRID_LENGTH);

	cudaMemcpyToArray(cuda_texBlockArrayT, 0, 0, pBlockCurrent->pucBlockExtendBuffer, iWidth*(iProcessSizeY + GRID_LENGTH), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(cuda_texPixel, cuda_texBlockArrayT, cuda_formatDescPixel);
	cuda_wallis_grid << <dimWallisGridBlocks, dimThreads >> >(cuda_pf2R0R1Buffer, iGridSizeX, iGridSizeY);

	cudaMemcpyToArray(cuda_texR0R1ArrayT, 0, 0, cuda_pf2R0R1Buffer, sizeof(float2)*CUDA_GRID_SIZE_X*iGridSizeY, cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(cuda_texR0R1, cuda_texR0R1ArrayT, cuda_formatDescR0R1);
	cuda_wallis_inner << <dimWallisInnerBlocks, dimThreads >> >(cuda_pucBlockOutputBuffer, iWidth);

	cudaUnbindTexture(cuda_texPixel);
	cudaUnbindTexture(cuda_texR0R1);
	cudaMemcpy(g_pucWriteBuffer, cuda_pucBlockOutputBuffer, iWidth*iProcessSizeY, cudaMemcpyDeviceToHost);

#ifdef _MSC_VER
	WaitForSingleObject(g_hMutexRasterIO, INFINITE);
	g_pBandOutput->RasterIO(GF_Write, 0, iPosY, iRasterSizeX, iProcessSizeY, g_pucWriteBuffer, iRasterSizeX, iProcessSizeY, GDT_Byte, 0, iWidth);
	ReleaseMutex(g_hMutexRasterIO);
#else
	sem_wait(&g_semRasterIO);
	g_pBandOutput->RasterIO(GF_Write, 0, iPosY, iRasterSizeX, iProcessSizeY, g_pucWriteBuffer, iRasterSizeX, iProcessSizeY, GDT_Byte, 0, iWidth);
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

	int iPointX = iXInGrid - cuda_GRID_HALF_LENGTH + iGridX*GRID_LENGTH;
	int iPointY = iYInGrid + iGridY*GRID_LENGTH;
	int iPixelValue = ((int)(iXInGrid&&iYInGrid))* ((int)tex2D(cuda_texPixel, iPointX, iPointY));

	iSumValidArray[iGridOffset] = (int)(iPixelValue > 0);
	iSumValueArray[iGridOffset] = iPixelValue;
	iSumSquareArray[iGridOffset] = iPixelValue*iPixelValue;
	__syncthreads();

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
			int iSumSquare = iSumSquareArray[0];
			float f32FilterMean = ((float)iSumValueArray[0]) / iValidPixelCount;
			float f32FilterVariance = (float)(iSumSquare / iValidPixelCount) + ((float)(iSumSquare%iValidPixelCount)) / iValidPixelCount - f32FilterMean*f32FilterMean;
			f32FilterVariance *= (float)(f32FilterVariance > 0.0f);
			float f32R1 = cuda_C_MUL_SIGMA_VALUE / (cuda_C_VALUE*sqrtf(f32FilterVariance) + cuda_1_C_MUL_SIGMA_VALUE);

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

	float f32PixelValue = (float)tex2D(cuda_texPixel, iPointX, iPointY + cuda_GRID_HALF_LENGTH);
	float2 f2R0R1 = tex2D(cuda_texR0R1, f32GridX, f32GridY);

	int iOutPixelValue = (int)((f2R0R1.y)*f32PixelValue + (f2R0R1.x) + 0.5f);
	iOutPixelValue -= ((int)(iOutPixelValue > 255)*(iOutPixelValue - 255));
	iOutPixelValue *= ((int)((iOutPixelValue > 0) && (f32PixelValue > 0.0f)));
	*(pucBlockOutputBuffer + (iPointY*iWidth + iPointX)) = (unsigned char)iOutPixelValue;
}