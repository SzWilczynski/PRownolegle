#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 8
#define DESTINATIONS 3
#define DENSITY 80
#define MINLENGTH 25
#define MAXLENGTH 150

int main(int argc, char** argv)
{
    srand(time(NULL));
    int size, destinations, density, minlength, maxlength;
    if(argc > 1)
        size = atoi(argv[1]);
    else
        size = SIZE;

    if(argc > 2)
        destinations = atoi(argv[2]);
    else
        destinations = DESTINATIONS;

    if(argc > 3)
        density = atoi(argv[3]);
    else
        density = DENSITY;

    if(argc > 4)
        minlength = atoi(argv[4]);
    else
        minlength = MINLENGTH;

    if(argc > 5)
        maxlength = atoi(argv[5]);
    else
        maxlength = MAXLENGTH;
    
    int** matrix = (int**)malloc(sizeof(int*)*size);
    bool* destination = (bool*)malloc(sizeof(bool)*size);
    for(int i = 0; i < size; i++)
    {
        matrix[i] = (int*)malloc(sizeof(int)*size);
        destination[i] = false;
    }

    for(int j = size-destinations; j < size; j++)
    {
        int t = rand()%j;
        if(!destination[t])
            destination[t] = true;
        else
            destination[j] = true;
    }

    for(int i = 0; i < size; i++){
        matrix[i][i] = 0;
        for(int y = i+1; y < size; y++){
            matrix[i][y] = (rand()%((maxlength-minlength)+1))+minlength;
            matrix[y][i] = matrix[i][y];
        }
    }

    for(int i = 0; i < size; i++)
        printf("%d", destination[i]);

    for(int i = 0; i < size; i++)
    {
        printf("\n");
        for(int y = 0; y < i; y++)
            printf("%d ", matrix[i][y]);
    }

    free(matrix);
    free(destination);
    return 0;
}
