#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <fstream>

#define DEV_MATRIX(mat, x, y, matSize) mat[(x)*matSize + (y)]

using namespace std;

long long int timePassed(struct timespec start, struct timespec end)
{
    long long int seconds = (end.tv_sec - start.tv_sec);
    long long int nanos = (end.tv_nsec - start.tv_nsec);
    return seconds*1000000000 + nanos;
}

__device__ int* DevAllocateMatrix(int matSize)
{
    return (int*)malloc(sizeof(int)*matSize*matSize);
}

__host__ int** AllocateMatrix(int matSize)
{
    int** matrix = (int**)malloc(sizeof(int*)*matSize);
    for(int i = 0; i < matSize; i++)
        matrix[i] = (int*)malloc(sizeof(int)*matSize);
    return matrix;
}


void FreeMatrix(int** matrix, int matSize)
{
    for(int i = 0; i < matSize; i++)
        free(matrix[i]);
    free(matrix);
}

void CudaAllocateMatrix(int** ptr, int matSize)
{
    cudaMalloc(ptr, sizeof(int)*matSize*matSize);
}

void CudaFreeMatrix(int* matrix)
{
    cudaFree(matrix);
}

void CudaMatrixUpload(int** input, int* output, int matSize)
{
    for(int x = 0; x < matSize; x++)
    {
        cudaMemcpy(output + (x*matSize), input[x], sizeof(int)*matSize, cudaMemcpyHostToDevice);
    }       
}

//Pawan Harish and P. J. Narayanan 2007 "Accelerating large graph algorithms on the GPU using CUDA"
__global__ void FloydWarshall(int* graphMatrix, int* secondGraphMatrix, int matSize, int iteration)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= matSize*matSize)
    {
        return;
    }
    int first = index/matSize;
    int second = index%matSize;
    if(first >= second || first == iteration || second == iteration)
    {
        return;
    }
    int firstAlt = DEV_MATRIX(graphMatrix, first, iteration, matSize);
    int secondAlt = DEV_MATRIX(graphMatrix, iteration, second, matSize);
    int newDist = firstAlt + secondAlt;
    if(firstAlt != 0
    && secondAlt != 0
    && (newDist < DEV_MATRIX(graphMatrix, first, second, matSize) || DEV_MATRIX(graphMatrix, first, second, matSize) == 0))
    {
        DEV_MATRIX(secondGraphMatrix, first, second, matSize) = newDist;
        DEV_MATRIX(secondGraphMatrix, second, first, matSize) = newDist;
    }else
    {
        DEV_MATRIX(secondGraphMatrix, first, second, matSize) = DEV_MATRIX(graphMatrix, first, second, matSize);
        DEV_MATRIX(secondGraphMatrix, second, first, matSize) = DEV_MATRIX(graphMatrix, first, second, matSize);
    }
}

__global__ void CompressOutput(int* graphMatrix, int matSize, int* destinations, int* outputPtr, int outputSize)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= matSize*matSize)
    {
        return;
    }
    int first = index/matSize;
    int second = index%matSize;
    if(first == second)
    {
        return;
    }
    int firstNode = destinations[first];
    int secondNode = destinations[second];
    if(firstNode == 0 || secondNode == 0)
    {
        return;
    }
    firstNode--;
    secondNode--;
    int dist = DEV_MATRIX(graphMatrix, first, second, matSize);

    DEV_MATRIX(outputPtr, firstNode, secondNode, outputSize) = dist;
    DEV_MATRIX(outputPtr, secondNode, firstNode, outputSize) = dist;

}

__global__ void GeneratePermutations(int* permutations, int permutationLength, int permutationCount, int offset)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index+offset >= permutationCount)
    {
        return;
    }

    int ogIndex = index;
    index += offset;

    int fact = 1;
    int lastMult;
    for(lastMult = 2; lastMult < permutationLength;)
    {
        fact *= lastMult;
        lastMult++;
    }
    lastMult--;

    for(int k = 0; k < permutationLength; k++)
    {
        DEV_MATRIX(permutations, ogIndex, k, permutationLength) = index/fact;
        index = index%fact;
        fact /= lastMult;
        lastMult--;
        if(lastMult < 1) lastMult = 1;
    }

    for(int k = permutationLength - 1; k > 0; k--)
    {
        for(int j = k-1; j>= 0; j--)
        {
            if(DEV_MATRIX(permutations, ogIndex, j, permutationLength) <= DEV_MATRIX(permutations, ogIndex, k, permutationLength))
            {
                DEV_MATRIX(permutations, ogIndex, k, permutationLength)++;
            }
        }
    }
}

__global__ void CalculateCycleLengths(int* graphMatrix, int* permutations, int matrixSize, int permutationCount, int* outputPtr, int offset, bool* isBest)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index+offset >= permutationCount)
    {
        return;
    }
    bool isValid = true;
    int sum = 0;
    int prev = 0;
    int daSize = matrixSize-1;
    for(int i = 0; i < daSize; i++)
    {
        int next = DEV_MATRIX(permutations, index, i, daSize)+1;
        if(DEV_MATRIX(graphMatrix, prev, next, matrixSize) == 0)
        {
            isValid = false;
        }
        sum += DEV_MATRIX(graphMatrix, prev, next, matrixSize);
        prev = next;
    }
    sum += DEV_MATRIX(graphMatrix, 0, prev, matrixSize);
    if(isValid)
    {
        outputPtr[index] = sum;
    }else{
        outputPtr[index] = 0;
    }
    isBest[index] = isValid;
}

__global__ void SelectSmallestResult(int* cycleLengths, bool* isShortest, int permLength)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= permLength)
    {
        return;
    }
    if(cycleLengths[index] == 0)
    {
        isShortest[index] = false;
        return;
    }
    int cycleLength = cycleLengths[index];
    for(int i = 0; i < permLength; i++)
    {
        if(cycleLengths[i] != 0 && (cycleLength > cycleLengths[i] || (cycleLength == cycleLengths[i] && index > i)))
        {
            isShortest[index] = false;
            return;
        }
    }
}

__global__ void ReturnSmallestResult(int* cycleLengths, bool* isShortest, int permLength, int* resultPtr)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= permLength)
    {
        return;
    }
    if(isShortest[index])
    {
        //printf("\nBest: %d %d\n", index, cycleLengths[index]);
        *resultPtr = cycleLengths[index];
    }
}

int Factorial(int val)
{
    int result = 1;
    for(int i = 2; i <= val; i++)
    {
        result *= i;
    }
    //printf("Factorial: %d %d\n", result, result*result);
    return result;
}


int main(int argc, char** argv)
{

    size_t real_size;
    int matSize;
    int destinationCount;

    int **graphMatrix;
    int *destinations;

    string line;
    ifstream inputf;

    if(argc < 2)
    {
        printf("Expected filepath to input data!\n");
        return 0;
    }
    inputf.open(argv[1]);
    if (!inputf.good())
    {
        printf("Failed to open file!\n");
        return 0;
    }

    getline(inputf, line);
    real_size = line.size();
    matSize = (int)real_size;
    destinationCount = 0;
    int matrixSize = matSize;
    graphMatrix = AllocateMatrix(matrixSize);
    destinations = (int*)malloc(sizeof(int)*matrixSize);

    for(int i = 0; i < matrixSize; i++)
        for(int y = 0; y < matrixSize; y++)
            graphMatrix[i][y] = 0;

    for(int i = 0; i < matrixSize; i++)
    {
        if(line[i] == '1')
        {
            destinationCount++;
            destinations[i] = destinationCount;
        }else
            destinations[i] = 0;
    }

    for(int point = 0; point < matrixSize; point++)
    {
        getline(inputf, line);
        real_size = line.size();
        matSize = (int)real_size;
        
        int connection = 0;
        for(int i = 0; i < matSize; i++)
        {
            if(line[i] != ' ')
            {
                graphMatrix[point][connection] *= 10;
                graphMatrix[point][connection] += (line[i] - '0');
            }else
            {
                graphMatrix[connection][point] = graphMatrix[point][connection];
                connection++;
            }
        }
    }


    inputf.close();
    for(int i = 0; i < matrixSize; i++)
    {
        for(int y = 0; y < matrixSize; y++)
        {
            //printf("%d, ", graphMatrix[i][y]);
        }
        //printf("\n");
    }

    int blockSize = 512;
    int blockCount;

    int result;
    int* devResult;
    struct timespec startTime, endTime;
    long long int timeElapsed;
    int* deviceMatrix;
    int* deviceMatrixCopy;
    int* transformedDeviceMatrix;
    int* deviceDestinations;

    cudaMalloc(&deviceDestinations, sizeof(int)*matrixSize);
    cudaMalloc(&devResult, sizeof(int));
    cudaMemcpy(deviceDestinations, destinations, sizeof(int)*matrixSize, cudaMemcpyHostToDevice);
    CudaAllocateMatrix(&deviceMatrix, matrixSize);
    CudaAllocateMatrix(&deviceMatrixCopy, matrixSize);
    CudaAllocateMatrix(&transformedDeviceMatrix, destinationCount);
    CudaMatrixUpload(graphMatrix, deviceMatrix, matrixSize);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceMatrixCopy, deviceMatrix, sizeof(int)*matrixSize*matrixSize, cudaMemcpyDeviceToDevice);

    FreeMatrix(graphMatrix, matrixSize);
    free(destinations);

    cudaDeviceSynchronize();
    timespec_get(&startTime, TIME_UTC);

    blockCount = 1+((matrixSize*matrixSize)/blockSize);

    int* mem;
    for(int i = 0; i < matrixSize; i++)
    {
        FloydWarshall<<<blockCount, blockSize>>>(deviceMatrix, deviceMatrixCopy, matrixSize, i);
        cudaDeviceSynchronize();
        
        mem = deviceMatrix;
        deviceMatrix = deviceMatrixCopy;
        deviceMatrixCopy = mem;
    }
    CompressOutput<<<blockCount, blockSize>>>(deviceMatrix, matrixSize, deviceDestinations, transformedDeviceMatrix, destinationCount);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    timespec_get(&endTime, TIME_UTC);
    timeElapsed = timePassed(startTime, endTime);
    //printf("[CUDA] Calculating distances time taken: %lld\n", timeElapsed);
    printf("%lld\n", timeElapsed);
    cudaFree(deviceMatrix);

    int possiblePermutations = Factorial(destinationCount-1);
    int bestResult = 0;

    int permutationSpace = 40960;

    int permutationCount = permutationSpace/(destinationCount-1);
    blockCount = permutationCount/blockSize;
    if(permutationCount%blockSize != 0) blockCount++;

    int* devicePermutations;
    bool* deviceIsBestPermutation;
    int* devicePermutationLengths;
    cudaMalloc(&devicePermutations, blockCount*blockSize*(destinationCount-1)*sizeof(int));
    cudaMalloc(&deviceIsBestPermutation, blockCount*blockSize*sizeof(bool));
    cudaMalloc(&devicePermutationLengths, blockCount*blockSize*sizeof(int));
    cudaDeviceSynchronize();

    timespec_get(&startTime, TIME_UTC);
    
    for(int i = 0; i < possiblePermutations; i += blockCount*blockSize)
    {
        cudaMemset(devicePermutations, 0, blockCount*blockSize*(destinationCount-1)*sizeof(int));
        cudaDeviceSynchronize();

        GeneratePermutations<<<blockCount, blockSize>>>(devicePermutations, destinationCount-1, possiblePermutations, i);
        cudaMemset(deviceIsBestPermutation, false, blockCount*blockSize*sizeof(bool));
        cudaMemset(devicePermutationLengths, 0, blockCount*blockSize*sizeof(int));
        cudaDeviceSynchronize();

        CalculateCycleLengths<<<blockCount, blockSize>>>(transformedDeviceMatrix, devicePermutations, destinationCount, possiblePermutations, devicePermutationLengths, i, deviceIsBestPermutation);
        cudaDeviceSynchronize();

        SelectSmallestResult<<<blockCount, blockSize>>>(devicePermutationLengths, deviceIsBestPermutation, permutationCount);
        cudaDeviceSynchronize();

        ReturnSmallestResult<<<blockCount, blockSize>>>(devicePermutationLengths, deviceIsBestPermutation, permutationCount, devResult);
        cudaDeviceSynchronize();

        cudaMemcpy(&result, devResult, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if(result < bestResult || bestResult == 0)
        {
            bestResult = result;
        }
    }

    timespec_get(&endTime, TIME_UTC);
    timeElapsed = timePassed(startTime, endTime);
    //printf("[CUDA] Brute force: %d time taken: %lld\n", bestResult, timeElapsed);
    printf("%lld\n", timeElapsed);

    cudaFree(deviceIsBestPermutation);
    cudaFree(devicePermutationLengths); 
    cudaFree(transformedDeviceMatrix);

    return 0;
}
