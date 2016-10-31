#ifdef MATRIXIO_H

template<class T>
int cuReadMatrixFromDisk(const char * filename, int * rows, int * cols, T ** matrix)
{
	FILE * file;
	fopen_s(&file, filename, "rb");
	if (!file)
	{
		printf("Can't open input matrix file: %s.\n", filename);
		return 1;
	}

	if (ReadMatrixSizeFromStream(file, rows, cols) != 0)
	{
		printf("Error reading matrix header from disk file: %s.\n", filename);
		return 1;
	}

	cudaHostAlloc((void**)&*matrix, sizeof(T)*(*rows)*(*cols), cudaHostAllocDefault);

	if (ReadMatrixFromStream(file, *rows, *cols, *matrix) != 0)
	{
		printf("Error reading matrix data from disk file: %s.\n", filename);
		return 1;
	}

	fclose(file);

	return 0;
}

template <class T>
int ReadMatrixFromDisk(const char * filename, int * rows, int * cols, T ** matrix)
{
	FILE * file;
	fopen_s(&file, filename, "rb");
	if (!file)
	{
		printf("Can't open input matrix file: %s.\n", filename);
		return 1;
	}

	if (ReadMatrixSizeFromStream(file, rows, cols) != 0)
	{
		printf("Error reading matrix header from disk file: %s.\n", filename);
		return 1;
	}

	//int size = (*m) * (*n) * sizeof(T) + 2 * sizeof(int);
	*matrix = (T *)malloc(sizeof(T)*(*rows)*(*cols));

	if (ReadMatrixFromStream(file, *rows, *cols, *matrix) != 0)
	{
		printf("Error reading matrix data from disk file: %s.\n", filename);
		return 1;
	}

	fclose(file);

	return 0;
}

template <class T>
int ReadMatrixFromStream(FILE * file, int rows, int cols, T * matrix)
{
	unsigned int readBytes;
	if ((readBytes = fread(matrix, sizeof(T), rows*cols, file)) < (unsigned int)rows*cols)
	{
		printf("Error: I have only read %u bytes. sizeof(T)=%lu\n", readBytes, sizeof(T));
		return 1;
	}

	return 0;
}

template <class T>
int WriteMatrixToDisk(const char * filename, int rows, int cols, T * matrix)
{
	FILE * file;
	fopen_s(&file, filename, "wb");
	if (!file)
	{
		printf("Can't open output file: %s.\n", filename);
		return 1;
	}

	if (WriteMatrixHeaderToStream(file, rows, cols) != 0)
	{
		printf("Error writing the matrix header to disk file: %s.\n", filename);
		return 1;
	}

	if (WriteMatrixToStream(file, rows, cols, matrix) != 0)
	{
		printf("Error writing the matrix to disk file: %s.\n", filename);
		return 1;
	}

	fclose(file);

	return 0;
}

template <class T>
int WriteMatrixToStream(FILE * file, int rows, int cols, T * matrix)
{
	if ((int)(fwrite(matrix, sizeof(T), rows*cols, file)) < rows*cols)
		return 1;
	return 0;
}

template <class T>
void PrintMatrixInMatlabFormat(int rows, int cols, T * U)
{
	printf("{\n");

	for (int i = 0; i < rows; i++)
	{
		printf("{");

		for(int j = 0; j < cols; j++)
		{
			printf("%f", U[ELT(cols, i, j)]);
			if(j != cols - 1)
				printf(", ");
		}
		printf("}");

		if(i != rows - 1)
			printf(",\n");
	}

	printf("\n}\n");
}


#endif