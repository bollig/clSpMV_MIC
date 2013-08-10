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
int RBFFD_IO<T>::loadFromBinaryEllpackFile(std::vector<int>& col_id, 
        int& nb_rows, int& stencil_size, std::string& filename)
{
    //printf("inside loadFromBinaryEllpackFile\n");
    char* line = (char*) malloc(255);
    FILE* fd = fopen(filename.c_str(), "rb");
    // comment line
    size_t nbchars = 250;
    getline(&line, &nbchars, fd);
    //printf("line= %s\n", line);
    getline(&line, &nbchars, fd);
    //printf("line= %s\n", line);
    sscanf(line, "%d %d", &nb_rows, &stencil_size);
    //printf("nb_rows= %d, stencil_size= %d\n", nb_rows, stencil_size);

    col_id.resize(nb_rows*stencil_size);
    int nb = fread(&col_id[0], sizeof(int), col_id.size(), fd);
    if (nb != col_id.size()) {
        printf("loadFromBinaryEllpackFile: nb elements read is incorrect\n");
        exit(0);
    }
    //for (int i=0; i < 100; i++) {
        //printf("col_id[%d]= %d\n", i, col_id[i]);
   //}
}
//--------------------------------------------------------------------
template <typename T>
int RBFFD_IO<T>::loadFromBinaryEllpackFileMulti(int& nb_subdomains, 
                std::vector<int>& col_id, std::vector<int>& nb_rows_multi, 
                std::vector<int>& nb_vec_elem_multi, std::vector<int>& offsets, 
                std::vector<int>& Qbeg, std::vector<int>& Qend, 
                int& stencil_size, std::string& filename)
{
    // the file contains multiple subdomains
    //printf("inside loadFromBinaryEllpackFile\n");
    char* line = (char*) malloc(255);
    FILE* fd = fopen(filename.c_str(), "rb");
    // comment line
    size_t nbchars = 250;
    getline(&line, &nbchars, fd);
    getline(&line, &nbchars, fd);
    sscanf(line, "%d", &nb_subdomains);

    offsets.resize(nb_subdomains+1);
    offsets[0] = 0;
    int max_rows = 0;

    for (int i=0; i < nb_subdomains; i++) {
        getline(&line, &nbchars, fd);
        int nb_rows, nb_vec;
        sscanf(line, "%d %d %d", &nb_rows, &stencil_size, &nb_vec);
        nb_rows_multi.push_back(nb_rows);
        nb_vec_elem_multi.push_back(nb_vec);
        offsets[i+1] = offsets[i] + nb_rows*stencil_size;
        max_rows += nb_rows;
    }

    col_id.resize(offsets[nb_subdomains]);
    int nb = fread(&col_id[0], sizeof(int), col_id.size(), fd);

    if (nb != col_id.size()) {
        printf("loadFromBinaryEllpackFileMulti: nb elements read is incorrect\n");
        exit(0);
    }

    printf("MAX NB ROWS (all subdomains combined): %d\n", max_rows);
    Qbeg.resize(max_rows);
    Qend.resize(max_rows);
    int nbb = fread(&Qbeg[0], sizeof(int), max_rows, fd);
    int nbe = fread(&Qend[0], sizeof(int), max_rows, fd);
    for (int i=0; i < 10; i++) {
        printf("io.rd; Qbeg,Qend= %d, %d\n", Qbeg[i], Qend[i]);
    }
    if (nbb != max_rows || nbe != max_rows) {
        printf("loadFromBinaryEllpackFileMulti: Qbeg or Qend read error\n");
        exit(0);
    }

    fclose(fd);
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

	printf("inside loadFromBinary: filename: %s\n", filename.c_str());

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

//printf("fromBinary, width/height= %d, %d\n", width, height);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	struct one_row_double {
		int row;
		int col;
		double val;
	} line_d;

	struct one_row_float {
		int row;
		int col;
		float val;
	} line_f;

	//printf("sizeof(line_d) = %d\n", sizeof(line_d));
	//printf("sizeof(line_f) = %d\n", sizeof(line_f));


	// Figure out the remaining size of the file to determine whether data is stored in 
	// float or double format. 
	
	int cur = ftell(fd); 
	rewind(fd);
	fseek(fd, 0, SEEK_END);
	int end = ftell(fd);
	rewind(fd);
	fseek(fd, cur, SEEK_SET); // not portable

	//printf("cur= %d, end= %d\n", cur, end);
	//printf("nonzeros= %d\n", nonzeros);
	int row_size = (end-cur) / nonzeros;
	int value_size = row_size - 2*sizeof(int);
	//printf("value_size= %d\n", value_size);

	if (value_size == sizeof(float)) {
		printf("File is stored as floats\n");
		for (int i=0; i < nonzeros; i++) {
			fread(&line_f, sizeof(line_f), 1, fd);
			rows[i] = line_f.row-1; // matrix market indexes from 1
			cols[i] = line_f.col-1;
			values[i] = line_f.val;
			//if (i < 10) printf("(float) fromBinary: %d, %d, %f\n", rows[i], cols[i], values[i]);
    	}
	} else if (value_size == sizeof(double)) {
		printf("File is stored as doubles\n");
		for (int i=0; i < nonzeros; i++) {
			fread(&line_d, sizeof(line_d), 1, fd);
			rows[i] = line_d.row-1; // matrix market indexes from 1
			cols[i] = line_d.col-1;
			values[i] = line_d.val;
			//if (i < 10) printf("(float) fromBinary: %d, %d, %lf\n", rows[i], cols[i], values[i]);
    	}
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

		// memory is not guaranteed to be contiguous which creates problems
		struct one_row {
			int row;
			int col;
			T val;
		} line;

        /* NOTE: matrix market files use 1-based indices, i.e. first element
           of a vector has index 1, not 0.  */

        for (unsigned int i = 0; i < nonzeros; i++) {
                // Add 1 because matrix market assumes we index 1:N instead of 0:N-1
				int r = rows[i] + 1;
				int c = cols[i] + 1;
				//line.row = rows[i]+1;
				//line.col = cols[i]+1;
				//line.val = values[i];
				//if (i < 10) printf("writeToBinary: %d, %d, %f\n", line.row, line.col, line.val);
                fwrite(&r, 1 , sizeof(int), fd);
                fwrite(&c, 1 , sizeof(int), fd);
                fwrite(&values[i], 1 , sizeof(T), fd);
        }

        fclose(fd);
		return(0);
}
//----------------------------------------------------------------------
