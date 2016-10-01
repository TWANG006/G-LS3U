#include "Utils.h"
#include <math.h>
#include <iostream>
#include <string>
#include <sstream>

namespace WFT_FPA
{
void fftwComplexMul(fftwf_complex& out, const fftwf_complex& in1, const fftwf_complex& in2)
{
	// Make a copy first to avoid self-assignment
	fftwf_complex temp;

	temp[0] = in1[0] * in2[0] - in1[1] * in2[1];
	temp[1] = in1[0] * in2[1] + in1[1] * in2[0];

	out[0] = temp[0];
	out[1] = temp[1];
}
void fftwComplexMul(fftw_complex& out, const fftw_complex& in1, const fftw_complex& in2)
{
	// Make a copy first to avoid self-assignment
	fftw_complex temp;

	temp[0] = in1[0] * in2[0] - in1[1] * in2[1];
	temp[1] = in1[0] * in2[1] + in1[1] * in2[0];

	out[0] = temp[0];
	out[1] = temp[1];
}

void fftwComplexScale(fftw_complex& out, const double s)
{
	out[0] = out[0] * s;
	out[1] = out[1] * s;
}
void fftwComplexScale(fftwf_complex& out, const float s)
{
	out[0] = out[0] * s;
	out[1] = out[1] * s;
}

float fftwComplexAbs(const fftwf_complex& in)
{
	return sqrt(in[0]*in[0] + in[1]*in[1]);
}
double fftwComplexAbs(const fftw_complex& in)
{
	return sqrt(in[0]*in[0] + in[1]*in[1]);
}

void fftwComplexPrint(const fftwf_complex& in)
{
	std::cout<<in[0]<<"+"<<"("<<in[1]<<"i)";
}
void fftwComplexPrint(const fftw_complex& in)
{
	std::cout<<in[0]<<"+"<<"("<<in[1]<<"i)";
}

bool fftwComplexMatRead2D(std::istream& in, fftwf_complex *&f, int& rows, int& cols)
{
	std::string line;					// Real line-by-line
	std::getline(in, line);				// Read the first line to get the size
	std::stringstream lineStream(line);	// read the line into stirng stream
	std::string cell;					// string of each cell in one line
	std::vector<int> size;				// size information
	std::vector<float> data;			// data(real,imag) of each line

	while (std::getline(lineStream, cell, ','))
	{
		size.push_back(std::stoi(cell));
	}
	// Check if the size is read already
	if(size.size() !=2)
		return false;
	rows = size[0];
	cols = size[1];

	// Allocate memory for f
	f = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * rows * cols);

	int i = 0;	// Index

	while (std::getline(in, line))
	{
		data.clear();
		std::stringstream lineStream(line);
		std::string cell;
		while(std::getline(lineStream, cell, ','))
		{
			data.push_back(std::stof(cell));
		}
		if(data.size() !=2)
			return false;

		f[i][0] = data[0];
		f[i][1] = data[1];
		
		i++;
	}

	return true;
	
}
bool fftwComplexMatRead2D(std::istream& in, fftw_complex *&f, int& rows, int& cols)
{
	std::string line;					// Real line-by-line
	std::getline(in, line);				// Read the first line to get the size
	std::stringstream lineStream(line);	// read the line into stirng stream
	std::string cell;					// string of each cell in one line
	std::vector<int> size;				// size information
	std::vector<double> data;			// data(real,imag) of each line

	while (std::getline(lineStream, cell, ','))
	{
		size.push_back(std::stoi(cell));
	}
	// Check if the size is read already
	if(size.size() !=2)
		return false;
	rows = size[0];
	cols = size[1];

	// Allocate memory for f
	f = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);

	int i = 0;	// Index

	while (std::getline(in, line))
	{
		data.clear();
		std::stringstream lineStream(line);
		std::string cell;
		while(std::getline(lineStream, cell, ','))
		{
			data.push_back(std::stod(cell));
		}
		if(data.size() !=2)
			return false;

		f[i][0] = data[0];
		f[i][1] = data[1];
		
		i++;
	}
	return true;
}

void fftwComplexMatWrite2D(std::ostream& out, fftwf_complex *f, const int rows, const int cols)
{
	// First line contains total rows & cols of the matrix
	out<<rows<<","<<cols<<"\n";
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			int id = y * cols + x;
			out<<f[id][0]<<","<<f[id][1]<<"\n";
		}
	}
}
void fftwComplexMatWrite2D(std::ostream& out, fftw_complex *f, const int rows, const int cols)
{
	// First line contains total rows & cols of the matrix
	out<<rows<<","<<cols<<"\n";
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			int id = y * cols + x;
			out<<f[id][0]<<","<<f[id][1]<<"\n";
		}
	}
}

} // WFT_FPA