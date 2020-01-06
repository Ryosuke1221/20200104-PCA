#include <pcl/visualization/cloud_viewer.h>
#include <iostream>

//#include <Eigen/Core>
#include "TimeString.h"

int main()
{
	typedef float Scalar;
	vector<vector<Scalar>> all_observation_vec_vec;

	//input file to vec_vec
	{
		string filename_;
		filename_ = "../ExPCA.csv";

		CTimeString time_;
		ifstream ifs_(filename_);
		string str_;
		int f_cnt = 0;
		if (ifs_.fail()) cout << "Error: file could not be read." << endl;
		else {
			//naraha
			while (getline(ifs_, str_)) {		//readed to string from file

				vector<Scalar> one_observation_vec;
				vector<int> find_vec = time_.find_all(str_, ",");

				one_observation_vec.push_back(stod(str_.substr(0, find_vec[0])));
				int s_pos = 0;
				while (s_pos < find_vec.size() - 1)
				{
					Scalar value_ = stod(str_.substr(find_vec[s_pos] + 1, find_vec[s_pos + 1] - (find_vec[s_pos] + 1)));
					one_observation_vec.push_back(value_);
					s_pos++;
				}
				one_observation_vec.push_back(stod(str_.substr(find_vec[s_pos] + 1, str_.size() - (find_vec[s_pos] + 1))));

				all_observation_vec_vec.push_back(one_observation_vec);
			}
			ifs_.close();
		}
	}

	//cout << "make .csv" << endl;
	//{
	//	std::string filename_save;
	//	filename_save = "../ExPCA.csv";
	//	std::ofstream ofs_save;
	//	ofs_save.open(filename_save, std::ios::out);
	//	for (int i = 0; i < all_observation_vec_vec.size(); i++)
	//	{
	//		for (int j = 0; j < all_observation_vec_vec[i].size(); j++)
	//		{
	//			ofs_save << all_observation_vec_vec[i][j];
	//			if (j < all_observation_vec_vec[i].size() - 1) ofs_save << ",";
	//		}
	//		ofs_save << endl;
	//	}
	//}

	const int rows_Matrix = all_observation_vec_vec.size();
	const int cols_Matrix = all_observation_vec_vec[0].size();

	//cout << "insert vecvec to matrix" << endl;
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Data_Matrix(rows_Matrix, cols_Matrix);
	for (int i = 0; i < rows_Matrix; i++)
		for (int j = 0; j < cols_Matrix; j++)
			Data_Matrix(i, j) = all_observation_vec_vec[i][j];
	cout << "Data_Matrix" << endl;
	cout << Data_Matrix << endl;
	//{
	//	int aa;
	//	cin >> aa;
	//}

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Cor_Matrix(cols_Matrix, cols_Matrix);
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Cor_Matrix2(cols_Matrix, cols_Matrix);
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y_Data_Matrix(rows_Matrix, cols_Matrix);
	//https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
	//https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Sample_covariance
	{
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Data_Matrix_mean_sample(cols_Matrix, 1);
		Data_Matrix_mean_sample = Data_Matrix.rowwise().sum() * 1. / ((Scalar)(rows_Matrix - 1));	//sample mean

		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Data_Matrix_demean(rows_Matrix, cols_Matrix);
		Data_Matrix_demean = Data_Matrix.colwise() - Data_Matrix_mean_sample;

		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Cov_Data_Matrix(cols_Matrix, cols_Matrix);
		Cov_Data_Matrix = Data_Matrix_demean.transpose() * Data_Matrix_demean;

		for (int i = 0; i < cols_Matrix; i++)
			for (int j = 0; j < cols_Matrix; j++)
				Cor_Matrix(i, j) = Cov_Data_Matrix(i, j) / sqrt(Cov_Data_Matrix(i, i)* Cov_Data_Matrix(j, j));

		for (int i = 0; i < rows_Matrix; i++)
			for (int j = 0; j < cols_Matrix; j++)
				Y_Data_Matrix(i, j) = Data_Matrix_demean(i, j) / sqrt(Cov_Data_Matrix(j, j));
		Cor_Matrix2 = Y_Data_Matrix.transpose() * Y_Data_Matrix;
		cout << "Correlation matrix" << endl;
		//cout << Cor_Matrix << endl;
		cout << Cor_Matrix2 << endl;
		//{
		//	int aa;
		//	cin >> aa;
		//}
	}

	cout << "Eigen values and vectors" << endl;
	Eigen::Matrix<Scalar, 1, Eigen::Dynamic> eigen_values(1, cols_Matrix);
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_vectors(cols_Matrix, cols_Matrix);
	{
		//https://forum.kde.org/viewtopic.php?f=74&t=110265
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > eig(Cor_Matrix2);

		//cout << "rows:" << eig.eigenvalues().rows() << endl;
		//cout << "cols:" << eig.eigenvalues().cols() << endl;
		//{
		//	int aa;
		//	cin >> aa;
		//}

		cout << "eigen_values" << endl;
		for (int i = 0; i < cols_Matrix; i++)
			eigen_values.col(i) = eig.eigenvalues().row(cols_Matrix - 1 - i);
		cout << eigen_values << endl;

		cout << "eigen_vectors(col)" << endl;
		for (int i = 0; i < cols_Matrix; i++)
			eigen_vectors.col(i) = eig.eigenvectors().col(cols_Matrix - 1 - i);
		cout << eigen_vectors << endl;

		//cotribution rate
		Eigen::Matrix<Scalar, 1, Eigen::Dynamic> eigen_cotribution_rate(1, cols_Matrix);
		Eigen::Matrix<Scalar, 1, Eigen::Dynamic> eigen_cumulative_cotribution_rate(1, cols_Matrix);
		{
			Scalar sum_eigen_values = 0.;
			Scalar sum_cumulative_eigen_values = 0.;
			for (int i = 0; i < eigen_values.cols(); i++)
				sum_eigen_values += eigen_values(0, i);

			for (int i = 0; i < eigen_values.cols(); i++)
				eigen_cotribution_rate(0, i) = eigen_values(0, i) / sum_eigen_values;
			for (int i = 0; i < eigen_values.cols(); i++)
			{
				sum_cumulative_eigen_values += eigen_cotribution_rate(0, i);
				eigen_cumulative_cotribution_rate(0, i) = sum_cumulative_eigen_values;
			}
		}
		cout << "cotribution rate" << endl;
		cout << eigen_cotribution_rate << endl;
		cout << " cumulative cotribution rate" << endl;
		cout << eigen_cumulative_cotribution_rate << endl;
		//{
		//	int aa;
		//	cin >> aa;
		//}
	}

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Z_Data_Matrix(rows_Matrix, cols_Matrix);
	Z_Data_Matrix = Y_Data_Matrix * eigen_vectors;
	cout << "Principal component" << endl;
	//cout << Z_Data_Matrix << endl;

	for (int i = 0; i < rows_Matrix; i++)
	{
		cout << "i:" << i << endl;
		cout << Data_Matrix.row(i) << endl;
		cout << Z_Data_Matrix.row(i) << endl;
	}
	//{
	//	int aa;
	//	cin >> aa;
	//}


	cout << endl;
	cout << "sort" << endl;
	//https://qiita.com/uchida106/items/502c86e620c39cf29413
	{
		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Z_Data_Matrix_sort(rows_Matrix, cols_Matrix);
		Z_Data_Matrix_sort = Z_Data_Matrix;
		vector<int> index_vec;
		for (int i = 0; i < rows_Matrix; i++) index_vec.push_back(i);

		int index_criterion;
		index_criterion = 0;//0,1,2,...
		cout << "principal component:" << index_criterion << endl;

		Scalar tmp;
		int tmp_index;
		for (int i = 0; i < rows_Matrix; i++) {
			for (int j = rows_Matrix - 1; j > i; j--) {
				if (Z_Data_Matrix_sort(j, index_criterion) > Z_Data_Matrix_sort(j - 1, index_criterion)) {
					tmp = Z_Data_Matrix_sort(j, index_criterion);
					tmp_index = index_vec[j];
					Z_Data_Matrix_sort(j, index_criterion) = Z_Data_Matrix_sort(j - 1, index_criterion);
					index_vec[j] = index_vec[j - 1];
					Z_Data_Matrix_sort(j - 1, index_criterion) = tmp;
					index_vec[j - 1] = tmp_index;
				}
			}
		}
		//show
		for (int i = 0; i < rows_Matrix; i++)
		{
			//index_vec[i]
			int index_ = index_vec[i];
			Scalar sum_ = Data_Matrix.row(index_)(0, 0)
				+ Data_Matrix.row(index_)(0, 1)
				+ Data_Matrix.row(index_)(0, 2)
				+ Data_Matrix.row(index_)(0, 3);
			cout << "i:" << index_ << endl;
			cout << Data_Matrix.row(index_) << endl;
			//cout << Z_Data_Matrix.row(index_) << endl;
			cout << "sum:";
			cout << sum_ << endl;
			cout << "ratio(math+science):";
			cout << (Data_Matrix.row(index_)(0, 1)
				+ Data_Matrix.row(index_)(0, 2)) / sum_ << endl;
			cout << "ratio(NL+Engligh):";
			cout << (Data_Matrix.row(index_)(0, 0)
				+ Data_Matrix.row(index_)(0, 3)) / sum_ << endl;
		}

		////save data
		//{
		//	//make vec_vec
		//	vector<vector<Scalar>> saved_data_vec_vec;
		//	for (int i = 0; i < rows_Matrix; i++)
		//	{
		//		//sort
		//		vector<Scalar> saved_data_vec;

		//		int index_ = index_vec[i];
		//		saved_data_vec.push_back(index_);
		//		for (int j = 0; j < Data_Matrix.cols(); j++)
		//			saved_data_vec.push_back(Data_Matrix(index_, j));
		//		saved_data_vec_vec.push_back(saved_data_vec);

		//		////PC
		//		//int index_ = i;
		//		//saved_data_vec.push_back(index_);
		//		//for (int j = 0; j < Data_Matrix.cols(); j++)
		//		//	saved_data_vec.push_back(Z_Data_Matrix(i, j));
		//		//saved_data_vec_vec.push_back(saved_data_vec);
		//	}
		//	//save into file
		//	{
		//		std::string filename_save;
		//		filename_save = "../ShowPCA_" + to_string(index_criterion) + "pc.csv";	//sort
		//		//filename_save = "../ShowPCA_pc.csv";	//PC
		//		std::ofstream ofs_save;
		//		ofs_save.open(filename_save, std::ios::out);
		//		for (int i = 0; i < saved_data_vec_vec.size(); i++)
		//		{
		//			for (int j = 0; j < saved_data_vec_vec[i].size(); j++)
		//			{
		//				ofs_save << saved_data_vec_vec[i][j];
		//				if (j < saved_data_vec_vec[i].size() - 1) ofs_save << ",";
		//			}
		//			ofs_save << endl;
		//		}
		//	}
		//}

	}

	
	return 0;
}