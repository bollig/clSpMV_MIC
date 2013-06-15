// Convert Matrix Market file from ascii to binary

#include "rbffd/rbffd_io.h"
#include <string>
#include <assert.h>
#include <vector>
#include <cmath>


#define VECT std::vector<T> 

//----------------------------------------------------------------------
int main()
{
	std::string filename = "../../examples/testdata/thirtytwo_million.mtx";
	//std::string filename = "../../examples/testdata/one_million.mtx";
	//std::string filename = "../../examples/testdata/mat65k.mtx";


	std::vector<int> rows;
	std::vector<int> cols;
	std::vector<double> values;

	std::vector<double> new_values;
	int nonzeros;
	int width;
	int height;

	RBFFD_IO<double> io;

	io.loadFromAsciMMFile(rows, cols, values, width, height, filename);  
	printf("read: width, height= %d, %d\n", width, height);
	printf("rows size: %d\n", rows.size());
	std::string file_binary = filename + 'b';
	for (int i=0; i < 10; i++) {
		printf("%d, %d, %f\n", rows[i], cols[i], values[i]);
	}
	io.writeToBinaryMMFile(rows, cols, values, width, height, file_binary);

	io.loadFromBinaryMMFile(rows, cols, new_values, width, height, file_binary);

	for (int count=0, i=0; i < values.size() && count < 10; i++) {
		double diff = fabs(values[i] - new_values[i]);
		if (diff > 1.e-5) {
			printf("read/write val: %f, %f\n", new_values[i], values[i]);
			count++;
		}
	}

	return(0);
}
//----------------------------------------------------------------------
#if 0
		int loadFromBinaryMMFile(std::vector<int>& rows, std::vector<int>& cols, 
				std::vector<T>& values,int& width, int& height, int& nonzeros, std::string& filename);

		int writeToBinaryMMFile(std::vector<int>& rows, std::vector<int>& cols, 
				std::vector<float>& values, int width, int height, std::string& filename);
#endif
