#ifndef __MAT_IO__H__
#define __MAT_IO__H__

#include "matrix_storage.h"

namespace spmv {

template <typename T>
void ReadMMF(char* filename, coo_matrix<int, T>* mat)
{
    FILE* infile = fopen(filename, "r");
    char tmpstr[100];
    char tmpline[1030];
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    bool ifreal = false;
    if (strcmp(tmpstr, "real") == 0)
        ifreal = true;
    bool ifsym = false;
    fscanf(infile, "%s", tmpstr);
    if (strcmp(tmpstr, "symmetric") == 0)
        ifsym = true;
    int height = 0;
    int width = 0;
    int nnz = 0;
    while (true)
    {
	fscanf(infile, "%s", tmpstr);
	if (tmpstr[0] != '%')
	{
	    height = atoi(tmpstr);
	    break;
	}
	fgets(tmpline, 1025, infile);
    }

    fscanf(infile, "%d %d", &width, &nnz);
    mat->matinfo.height = height;
    mat->matinfo.width = width;
    int* rows = (int*)malloc(sizeof(int)*nnz);
    int* cols = (int*)malloc(sizeof(int)*nnz);
    T* data = (T*)malloc(sizeof(T)*nnz);
    int diaCount = 0;
    for (int i = 0; i < nnz; i++)
    {
        int rowid = 0;
        int colid = 0;
        fscanf(infile, "%d %d", &rowid, &colid);
        rows[i] = rowid - 1;
        cols[i] = colid - 1;
        data[i] = 1.0f;
        if (ifreal)
        {
            double dbldata = 0.0f;
            fscanf(infile, "%lf", &dbldata);
            data[i] = (T)dbldata;
        }
        if (rows[i] == cols[i])
            diaCount++;
    }
    
    if (ifsym)
    {
        int newnnz = nnz * 2 - diaCount;
        mat->matinfo.nnz = newnnz;
        mat->coo_row_id = (int*)malloc(sizeof(int)*newnnz);
        mat->coo_col_id = (int*)malloc(sizeof(int)*newnnz);
        mat->coo_data = (T*)malloc(sizeof(T)*newnnz);
        int matid = 0;
        for (int i = 0; i < nnz; i++)
        {
            mat->coo_row_id[matid] = rows[i];
            mat->coo_col_id[matid] = cols[i];
            mat->coo_data[matid] = data[i];
            matid++;
            if (rows[i] != cols[i])
            {
                mat->coo_row_id[matid] = cols[i];
                mat->coo_col_id[matid] = rows[i];
                mat->coo_data[matid] = data[i];
                matid++;
            }
        }
        assert(matid == newnnz);
        bool tmp = sort_coo<int, T>(mat);
        assert(tmp == true);
    }
    else
    {
        mat->matinfo.nnz = nnz;
        mat->coo_row_id = (int*)malloc(sizeof(int)*nnz);
        mat->coo_col_id = (int*)malloc(sizeof(int)*nnz);
        mat->coo_data = (T*)malloc(sizeof(T)*nnz);
        memcpy(mat->coo_row_id, rows, sizeof(int)*nnz);
        memcpy(mat->coo_col_id, cols, sizeof(int)*nnz);
        memcpy(mat->coo_data, data, sizeof(T)*nnz);
        if (!if_sorted_coo<int, T>(mat))
            sort_coo<int, T>(mat);
        //assert(if_sorted_coo(mat) == true);
    }
    
    fclose(infile);
    free(rows);
    free(cols);
    free(data);
}

};

#endif
