#if 0
#define ONE_MONOMIAL 1
#define SCALE_BY_H 0
#define SCALE_OUT_BY_H 0
// Default: 0
#define SCALE_B_BY_GAMMA 0
#endif


#include <stdio.h>
#include <vector>
#include <iostream>
//#include "nist/mmio.h"
#include "mmio.h"
#include <assert.h>


//#include "rbffd/stencils.h"
//#include "utils/geom/cart2sph.h"

//#include "rbffd.h"
//#include "rbfs/rbf_gaussian.h"
// For writing weights in (sparse) matrix market format

template <typename T>
int RBFFD_IO<T>::loadFromAsciMMFile(std::vector<int>& rows, std::vector<int>& cols, 
	std::vector<T>& values, int& width, int& height, std::string& filename)
{
	int ret_code;
	MM_typecode matcode;
	FILE *fd;
	int M, N;
	int nonzeros;

	if ((fd = fopen(filename.c_str(), "r")) == NULL) {
		std::cout << "File not found: " << filename << std::endl;
		return 1;
	}

	if (mm_read_banner(fd, &matcode) != 0) {
		std::cout << "Could not process MatrixMarket Banner in " << filename << std::endl;
		return 2;
	}

	if ((ret_code = mm_read_mtx_crd_size(fd, &M, &N, &nonzeros)) != 0) {
		std::cout << "Error! failed to parse file contents" << std::endl;
        return 4;
	}

	rows.resize(nonzeros);
	cols.resize(nonzeros);
	values.resize(nonzeros);

	width = N;
	height = M;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	int* row = &rows[0];
	int* col = &cols[0];
	T*   val = &values[0];

	for (int i=0; i < nonzeros; i++) {
        fscanf(fd, "%d %d %le\n", row+i, col+i, val+i);
		row[i]--;
		col[i]--;
		if (i < 10) printf("fromAsci: %d, %d, %f\n", rows[i], cols[i], values[i]);
    }

	fclose(fd);
	return(0);
}
//--------------------------------------------------------------------
template <typename T>
int RBFFD_IO<T>::loadFromBinaryMMFile(std::vector<int>& rows, std::vector<int>& cols, 
		std::vector<T>& values, int& width, int& height, std::string& filename)
{
	int ret_code;
	MM_typecode matcode;
	FILE *fd;
	int M, N;
	int nonzeros;

	if ((fd = fopen(filename.c_str(), "r")) == NULL)
	{
		std::cout << "File not found: " << filename << std::endl;
		return 1;
	}
	if (mm_read_banner(fd, &matcode) != 0)
	{
		std::cout << "Could not process MatrixMarket Banner in " << filename << std::endl;
		return 2;
	}

	if ((ret_code = mm_read_mtx_crd_size(fd, &M, &N, &nonzeros)) != 0)
	{
		std::cout << "Error! failed to parse file contents" << std::endl;
        return 4;
	}

	rows.resize(nonzeros);
	cols.resize(nonzeros);
	values.resize(nonzeros);

	width = N;
	height = M;

printf("fromBinary, width/height= %d, %d\n", width, height);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	struct one_row {
		int row;
		int col;
		T val;
	} line;

	printf("sizeof(line) = %d\n", sizeof(line));

	for (int i=0; i < nonzeros; i++) {
		fread(&line, sizeof(line), 1, fd);
		rows[i] = line.row-1; // matrix market indexes from 1
		cols[i] = line.col-1;
		values[i] = line.val;
		if (i < 10) printf("fromBinary: %d, %d, %f\n", rows[i], cols[i], values[i]);
    }

	fclose(fd);
	return(0);
}
//--------------------------------------------------------------------
template <typename T>
int RBFFD_IO<T>::writeToBinaryMMFile(std::vector<int>& rows, std::vector<int>& cols, 
	std::vector<T>& values, int width, int height, std::string& filename) 
{
		int M = height;
		int N = width;
		assert(rows.size() == cols.size());
		assert(rows.size() == values.size());

        // number of non-zeros (should be close to max_st_size*num_stencils)
		int nonzeros = rows.size();

        // Value obtained from mm_set_* routine
        MM_typecode matcode;

        //  int I[nz] = { 0, 4, 2, 8 };
        //  int J[nz] = { 3, 8, 7, 5 };
        //  double val[nz] = {1.1, 2.2, 3.2, 4.4};

        int err = 0;
        FILE* fd;
        fd = fopen(filename.c_str(), "w");
        err += mm_initialize_typecode(&matcode);
        err += mm_set_matrix(&matcode);
        err += mm_set_coordinate(&matcode);
        err += mm_set_real(&matcode);

        err += mm_write_banner(fd, matcode);
        err += mm_write_mtx_crd_size(fd, M, N, nonzeros);

		struct one_row {
			int row;
			int col;
			T val;
		} line;

        /* NOTE: matrix market files use 1-based indices, i.e. first element
           of a vector has index 1, not 0.  */

        for (unsigned int i = 0; i < nonzeros; i++) {
                // Add 1 because matrix market assumes we index 1:N instead of 0:N-1
				line.row = rows[i]+1;
				line.col = cols[i]+1;
				line.val = values[i];
				if (i < 10) printf("toBinary: %d, %d, %f\n", line.row, line.col, line.val);
                fwrite(&line, 1 , sizeof(line), fd);
        }

        fclose(fd);
		return(0);
}
//----------------------------------------------------------------------
