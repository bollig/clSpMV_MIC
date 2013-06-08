#ifndef _PRINT_UTIL_H_
#define _PRINT_UTIL_H_

#include "matrix_storage.h"

namespace spmv {

template <typename T>
void printDense(coo_matrix<int, T>& mat)
{
    int width = mat.matinfo.width;
    int height = mat.matinfo.height;
    T* dense = (T*)malloc(sizeof(T)*width*height);
    memset(dense, 0, sizeof(int)*width*height);
    for (int i = 0; i < mat.matinfo.nnz; i++)
    {
        int row = mat.coo_row_id[i];
        int col = mat.coo_col_id[i];
        T data = mat.coo_data[i];
        dense[row * width + col] = data;
    }
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%f ", dense[i * width + j]);
        }
        printf("\n");
    }
}
//----------------------------------------------------------------------

template <typename T>
void printCOO(coo_matrix<int, T>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    int nnz = mat.matinfo.nnz;
    for (int i =0 ; i < nnz; i++)
    {
        printf("row %d col %d data %f\n", mat.coo_row_id[i], mat.coo_col_id[i], mat.coo_data[i]);
    }
}
//----------------------------------------------------------------------

template <typename T>
void printDIA(dia_matrix<int, int, T>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    printf("dianum %d length %d alignLength %d\n", mat.dia_num, mat.dia_length, mat.dia_length_aligned);
    int num = mat.dia_num;
    printf("Offset: ");
    for (int i = 0; i < num; i++)
        printf("%d ", mat.dia_offsets[i]);
    printf("\n");
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < mat.dia_length_aligned; j++)
            printf("%f ", mat.dia_data[i * mat.dia_length_aligned + j]);
        printf("\n");
    }
}
//----------------------------------------------------------------------

template <typename T>
void printDIAext(dia_ext_matrix<int, int, T>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    printf("dianum %d length %d alignLength %d\n", mat.dia_num, mat.dia_length, mat.dia_length_aligned);
    int num = mat.dia_num;
    printf("Offset: ");
    for (int i = 0; i < num; i++)
        printf("%d ", mat.dia_offsets[i]);
    printf("\n");
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < mat.dia_length_aligned; j++)
            printf("%f ", mat.dia_data[i * mat.dia_length_aligned + j]);
        printf("\n");
    }
}
//----------------------------------------------------------------------

template <typename T>
void printCSR(csr_matrix<int, T>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    int nnz = mat.matinfo.nnz;
    int height = mat.matinfo.height;
    printf("Row ptr: ");
    for (int i = 0; i <= height; i++)
        printf("%d ", mat.csr_row_ptr[i]);
    printf("\n");
    printf("Col: ");
    for (int i = 0; i < nnz; i++)
        printf("%d ", mat.csr_col_id[i]);
    printf("\n");
    printf("Data: ");
    for (int i = 0; i < nnz; i++)
        printf("%f ", mat.csr_data[i]);
    printf("\n");
}
//----------------------------------------------------------------------

template <typename T>
void printBCSR(bcsr_matrix<int, T>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    printf("bwidth %d bheight %d rownum %d alignsize %d blocknum %d\n", mat.bcsr_bwidth, mat.bcsr_bheight, mat.bcsr_row_num, mat.bcsr_aligned_size, mat.bcsr_block_num);
    printf("Row ptr: ");
    for (int i = 0; i <= mat.bcsr_row_num; i++)
	printf("%d ", mat.bcsr_row_ptr[i]);
    printf("\n");
    printf("Col id: ");
    for (int i = 0; i < mat.bcsr_block_num; i++)
	printf("%d ", mat.bcsr_col_id[i]);
    printf("\n");
    printf("Data: \n");
    int blocksize = mat.bcsr_bwidth * mat.bcsr_bheight;
    for (int i = 0; i < blocksize; i++)
    {
	for (int j = 0; j < mat.bcsr_aligned_size; j++)
	    printf("%f ", mat.bcsr_data[j + i * mat.bcsr_aligned_size]);
	printf("\n");
    }
    printf("\n");
    
}
//----------------------------------------------------------------------
template <typename T>
void writeCSR(char* filename, csr_matrix<int, T>& mat)
{
	FILE* f = fopen(filename, "wb");
	unsigned int tmp = mat.matinfo.width;
	fwrite(&tmp, sizeof(unsigned int), 1, f);

	tmp = mat.matinfo.height;
    //read rows (UINT)
    fwrite(&tmp, sizeof(unsigned int), 1, f);

	tmp = mat.matinfo.nnz;
    //read # nonzero values (UINT)
    fwrite(&tmp, sizeof(unsigned int), 1, f);

    //read column indices (UINT *)
	fwrite(mat.csr_col_id, sizeof(unsigned int), mat.matinfo.nnz, f);

    //read row pointer (UINT *)
	fwrite(mat.csr_row_ptr, sizeof(unsigned int), mat.matinfo.height + 1, f);

    //read all nonzero values (T *)
	fwrite(mat.csr_data, sizeof(T), mat.matinfo.nnz, f);

    fclose(f);
}
//----------------------------------------------------------------------

};



#endif
