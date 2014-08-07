#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <stdint.h>
#include <Windows.h>

// # UNITS OF HIDDEN LAYER = # INPUTS TO TRAIN THE SYSTEM
#define HIDDEN   4
#define INPUT    2
#define OUTPUT   2

#define EPOCHS   300
#define PRECISION 0.01


using namespace std;

class CounterpropNetwork
{

#pragma region Attributes

private: 

	DWORD  m_dwInputSize;
	DWORD  m_dwHiddenSize;
	DWORD  m_dwOutputSize;
	double m_dbKohonenLearningRate;
	double m_dbKohonenDecayRate;
	double m_dbGrossbergLearningRate;
	double m_dbGrossbergDecayRate;
	double m_dbKohonenInitRange;
	double m_dbInhibitingFactor;
	DWORD  m_dwNumberOfWinnersAllowed;

public:

	DWORD  m_dwNumberOfIterations;
	double m_dbTolerance;
	double m_dbNoiseFactor;

	vector<vector<double>> m_vvdKohonenMatrix;
	vector<vector<double>> m_vvdGrossbergMatrix;
	vector<int>            m_viNumberOfWins;

#pragma endregion

#pragma region Methods

private:

	void updateKohonenWeights( int hiddenIndex, vector<double>& inputVector );
	void updateGrossbergWeights( DWORD hiddenIndex, double hiddenActivation, vector<double>& outputVector );
	int  getWinningHiddenIndex( vector<double>& vd_inputVector);
	void normalize( vector<double>& vd_inputVector );

public:

	CounterpropNetwork();
	~CounterpropNetwork();
	double         random();
	void           training( vector<vector<double>> vvd_trainingInputVector, vector<vector<double>> vvd_trainingOutputVector );
	int            trainingVector( vector<double>& inputVector );
	vector<double> testing( vector<double>& inputVector );
	void           printMatrix();

#pragma endregion

};