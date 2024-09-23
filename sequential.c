#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

long long int timePassed(struct timespec start, struct timespec end)
{
    long long int seconds = (end.tv_sec - start.tv_sec);
    long long int nanos = (end.tv_nsec - start.tv_nsec);
    return seconds*1000000000 + nanos;
}

int** AllocateMatrix(int size)
{
    int** matrix = (int**)malloc(sizeof(int*)*size);
    for(int i = 0; i < size; i++)
        matrix[i] = (int*)malloc(sizeof(int)*size);
    return matrix;
}

void FreeMatrix(int** matrix, int size)
{
    for(int i = 0; i < size; i++)
        free(matrix[i]);
    free(matrix);
}

void swap(int* first, int* second)
{
    int swapped = *first;
    *first = *second;
    *second = swapped;
}

int **Dijkstra(int **graphMatrix, int size, int destinationCount, bool *destinations)
{
    int** destinationDistances = AllocateMatrix(destinationCount);
    int* distances = (int*)malloc(sizeof(int)*size);
    bool* visited = (bool*)malloc(sizeof(bool)*size);
    int currentDestination = -1;
    for(int i = 0; i < size; i++)
    {
        //for each destination, run Dijkstra's algorithm to find shortest path to other destinations.
        if (destinations[i])
        {
            currentDestination++;
            destinationDistances[currentDestination][currentDestination] = 0;

            for(int n = 0; n < size; n++) //Set up starting values.
            {
                distances[n] = graphMatrix[i][n];
                visited[n] = false;
            }

            distances[i] = 0;
            visited[i] = true;

            int currentNode = i;
            for(;;){
                //Find next unvisited node with shortest distance.
                for(int n = 0; n < size; n++)
                    if(distances[n] != 0 && !visited[n] && (visited[currentNode] || distances[n] < distances[currentNode]))
                        currentNode = n;

                if(visited[currentNode]) break;  //End the loop if no unvisited node was found.
                visited[currentNode] = true;

                    //Assign new distance values to its neighbours.
                for(int n = 0; n < size; n++)
                    if(!visited[n]
                    && graphMatrix[currentNode][n] > 0
                    && ((distances[n] == 0 && n != i) || (distances[n] > distances[currentNode]+graphMatrix[currentNode][n]))
                    )
                        distances[n] = distances[currentNode]+graphMatrix[currentNode][n];
                        
            }

            //Get lengths between destinations and put them into the new graph.
            int n = currentDestination;
            for(int y = i+1; y < size; y++)
                if(destinations[y])
                {
                    n++;
                    destinationDistances[currentDestination][n] = distances[y];
                    destinationDistances[n][currentDestination] = destinationDistances[currentDestination][n];
                }
        }
    }
    free(distances);
    free(visited);

    return destinationDistances;
}

int cycleLength(int **graphMatrix, int size, int *permutation)
{
    int result = 0;
    for(int i = size-1; i > 0; i--)
    {
        result += graphMatrix[permutation[i]][permutation[i-1]];
    }
    result += graphMatrix[permutation[0]][permutation[size-1]];
    return result;
}

bool next_permutation(int* start, int length)
{
    int k, l;
    for (int i = length-2; ; i--)
    {
        if(i < 0) return false;
        if(start[i] < start[i+1])
        {
            k = i;
            break;
        }
    }
    for (int i = length-1; i > k; i--)
    {
        if(i == 0) return false;
        if(start[i] > start[k])
        {
            l = i;
            break;
        }
    }
    swap(&start[l], &start[k]);
    for(int i = 0; length-1-i > k+1+i ; i++)
    {
        swap(&start[length-1-i], &start[k+1+i]);
    }
    return true;
}

int bruteForce(int **graphMatrix, int size)
{
    int *permutation = (int*)malloc(sizeof(int)*(size));
    for(int i = 0 ; i < size; i++)
        permutation[i] = i;

    int result = cycleLength(graphMatrix, size, permutation);
    while(next_permutation(&permutation[1], size-1))
    {
        int new = cycleLength(graphMatrix, size, permutation);
        if(new < result) result = new;
    }

    free(permutation);
    return result;
}

int main(int argc, char** argv)
{
    int i, y, point, connection;

    size_t real_size;
    int size;
    int destinationCount;

    int **graphMatrix;
    bool *destinations;

    char* line = NULL;
    FILE* inputf = NULL;

    if(argc < 2)
    {
        printf("Expected filepath to input data!\n");
        return 0;
    }
    inputf = fopen(argv[1], "r");
    if (!inputf)
    {
        printf("Failed to open file!\n");
        return 0;
    }

    getline(&line, &real_size, inputf);
    real_size = strlen(line)-2;
    size = (int)real_size;
    destinationCount = 0;
    int matrixSize = size;
    graphMatrix = AllocateMatrix(matrixSize);
    destinations = (bool*)malloc(sizeof(bool)*matrixSize);

    for(i = 0; i < matrixSize; i++)
        for(y = 0; y < matrixSize; y++)
            graphMatrix[i][y] = 0;

    for(i = 0; i < size; i++)
    {
        if(line[i] == '1')
        {
            destinations[i] = true;
            destinationCount++;
        }else
            destinations[i] = false;
    }

    for(point = 0; point < matrixSize; point++)
    {
        free(line);
        line = NULL;
        getline(&line, &real_size, inputf);
        real_size = strlen(line)-2;
        size = (int)real_size;
        
        connection = 0;
        for(i = 0; i < size+5; i++)
        {
            if(line[i] == '\0')
            {
                break;
            }else if(line[i] != ' ')
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
    fclose(inputf);

    int result;
    struct timespec startTime, endTime;
    long long int timeElapsed;

    clock_gettime(CLOCK_MONOTONIC, &startTime);
    int **destinationDistances = Dijkstra(graphMatrix, matrixSize, destinationCount, destinations);
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    timeElapsed = timePassed(startTime, endTime);
    //printf("[SEQ] Calculating distances time taken: %lld\n", timeElapsed);
    printf("%lld\n", timeElapsed);

    FreeMatrix(graphMatrix, matrixSize);
    free(destinations);

    srand(time(NULL));  //Setting up the seed for RNG.

    clock_gettime(CLOCK_MONOTONIC, &startTime);
    result = bruteForce(destinationDistances, destinationCount);
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    timeElapsed = timePassed(startTime, endTime);
    //printf("[SEQ] Brute force: %d time taken: %lld\n", result, timeElapsed);
    printf("%lld\n", timeElapsed);

    FreeMatrix(destinationDistances, destinationCount);

    return 0;
}
