#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

unsigned int matrixSizeFinding(FILE* inputFile)
{
	unsigned int matrixSize;

	if (fscanf(inputFile, "%u", &matrixSize) < 1)
		return 0;

	return matrixSize;
}

char readFile(FILE* inputFile, float* matrix, unsigned int matrixSize)
{
	matrixSize *= matrixSize;

	for (unsigned int i = 0; i < matrixSize; i++)
	{
		if (fscanf(inputFile, "%f", &matrix[i]) <= 0)
			return 1;
	}
	return 0;
}

char referenceElementReplacement(float* matrix, unsigned int matrixSize, unsigned int k)
{
	for (unsigned int i = k + 1; i < matrixSize; i++)
	{
		unsigned int currLine = k * matrixSize;
		unsigned int replacementLine = i * matrixSize;

		if (fabs(matrix[replacementLine + i - 1]) >= 0.00001f)
		{
			for (unsigned int j = 0; j < matrixSize; j++)
			{
				float buff = matrix[currLine + j];
				matrix[currLine + j] = matrix[replacementLine + j];
				matrix[replacementLine + j] = buff * -1;
			}

			return 0;
		}
	}

	return 1;
}

char writeFile(FILE* outputFile, float determinant)
{
	if (fprintf(outputFile, "%f ", determinant) < 0)
		return 1;
	if (fprintf(outputFile, "%c", (char)10) < 0)
		return 1;

	return 0;
}

float determinantCalculating(float* matrix, unsigned int matrixSize)
{
	float determinant = 1;
	unsigned int lastElementOfFirstRowIndex = matrixSize - 1;

	for (unsigned int k = 0; k < lastElementOfFirstRowIndex; k++)
	{
		unsigned int refLine = k * matrixSize;
		unsigned int currPivotIndex = refLine + k;

		if (fabs(matrix[currPivotIndex]) <= 0.00001f)
		{
			if (referenceElementReplacement(matrix, matrixSize, k) == 1)
				return 0;
		}

		for (unsigned int i = k + 1; i < matrixSize; i++)
		{
			unsigned int currLine = i * matrixSize;
			float modifier = matrix[currLine + k] / matrix[refLine + k];

			for (unsigned int j = k; j < matrixSize; j++)
			{
				matrix[currLine + j] = matrix[currLine + j] - matrix[refLine + j] * modifier;
			}
		}

		determinant *= matrix[currPivotIndex];
	}

	return determinant * matrix[matrixSize * matrixSize - 1];
}

float parallelDeterminantCalculating(float* matrix, unsigned int matrixSize, short int* actualNumberOfThreads)
{
	unsigned int numOfThreads = *actualNumberOfThreads;
	unsigned int lastElementIndex = matrixSize - 1;
	unsigned int cutOffPoint = 7000 * (numOfThreads - 1); // целесообразность создания потоков

	for (unsigned int k = 0; k < lastElementIndex; k++)
	{
		unsigned int refLine = k * matrixSize;
		unsigned int currPivotIndex = refLine + k;

		if (fabs(matrix[currPivotIndex]) <= 0.00001f)
		{
			if (referenceElementReplacement(matrix, matrixSize, k) == 1)
				return 0;
		}

		if (((matrixSize - k) * (lastElementIndex - k) > cutOffPoint) && (lastElementIndex - k >= numOfThreads))
		{
#pragma omp parallel num_threads(numOfThreads) shared(matrix, refLine, k)
			{
#pragma omp for schedule(static, 20)
				for (unsigned int i = k + 1; i < matrixSize; i++)
				{
					unsigned int currLine = i * matrixSize;
					float modifier = matrix[currLine + k] / matrix[refLine + k];

					for (unsigned int j = k; j < matrixSize; j++)
						matrix[currLine + j] = matrix[currLine + j] - matrix[refLine + j] * modifier;
				}
			}
		}
		else
		{
			for (unsigned int i = k + 1; i < matrixSize; i++)
			{
				unsigned int currLine = i * matrixSize;
				float modifier = matrix[currLine + k] / matrix[refLine + k];

				for (unsigned int j = k; j < matrixSize; j++)
					matrix[currLine + j] = matrix[currLine + j] - matrix[refLine + j] * modifier;
			}
		}
	}

	float determinant = 1;
#pragma omp parallel num_threads(numOfThreads)
	{
		float t_determinant = 1;

#pragma omp for schedule(static, 100)
		for (int i = 0; i < matrixSize; i++)
			t_determinant *= matrix[i * matrixSize + i];

#pragma omp atomic
		determinant *= t_determinant;

#pragma omp single
		* actualNumberOfThreads = omp_get_num_threads();
	}

	return determinant;
}

int main(int argc, char* argv[])
{
	if (argc == 4)
	{
		short int numOfThreads = atoi(argv[3]);

		if (numOfThreads < -1)
		{
			fprintf(stderr, "Incorrect number of threads!\n");
			return 1;
		}

		FILE* inputFile = fopen(argv[1], "r");
		if (inputFile == NULL) {
			fprintf(stderr, "Input file open error!\n");
			return 1;
		}

		FILE* outputFile = fopen(argv[2], "w");
		if (outputFile == NULL) {
			fprintf(stderr, "Output file open error!\n");
			fclose(inputFile);
			return 1;
		}

		if (numOfThreads < -1)
		{
			fprintf(stderr, "Incorrect number of threads!\n");
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		unsigned short int maxThreads = omp_get_max_threads();

		if (numOfThreads == 0)
			numOfThreads = maxThreads;

		unsigned int matrixSize = matrixSizeFinding(inputFile);
		if (matrixSize < 1)
		{
			fprintf(stderr, "Invalid matrix size!\n");
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		float* matrix = (float*)malloc(sizeof(float) * matrixSize * matrixSize);
		if (matrix == NULL)
		{
			fprintf(stderr, "Insufficient memory available!\n");
			free(matrix);
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		if (readFile(inputFile, matrix, matrixSize))
		{
			fprintf(stderr, "Invalid file format!\n");
			free(matrix);
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		float determinant;
		double startRuntime, endRuntime;
		unsigned short int actualNumberOfThreads = 1;

		if (matrixSize == 1)
		{
			startRuntime = omp_get_wtime();
			determinant = matrix[0];
			endRuntime = omp_get_wtime();
		}
		else if (numOfThreads == -1)
		{
			startRuntime = omp_get_wtime();
			determinant = determinantCalculating(matrix, matrixSize);
			endRuntime = omp_get_wtime();
		}
		else
		{
			startRuntime = omp_get_wtime();
			determinant = parallelDeterminantCalculating(matrix, matrixSize, &numOfThreads);
			endRuntime = omp_get_wtime();
			actualNumberOfThreads = numOfThreads;
		}

		printf("Time (%i thread(s)): %g ms\n", actualNumberOfThreads, (endRuntime - startRuntime) * 1000);

		if (writeFile(outputFile, determinant))
		{
			fprintf(stderr, "File write error!\n");
			free(matrix);
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		free(matrix);
		fclose(inputFile);
		fclose(outputFile);
	}
	else
	{
		fprintf(stderr, "Not enough arguments!\n");
		return 1;
	}
}
