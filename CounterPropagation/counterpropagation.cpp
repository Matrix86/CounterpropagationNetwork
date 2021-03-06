#include "counterpropagation.h"

double CounterpropNetwork::random()
{
	double random_value;

	random_value = ( (double)( rand() % 100 ) ) / 101.0;

	return random_value;
}

CounterpropNetwork::CounterpropNetwork()
{
	m_dbGrossbergLearningRate   = 0.05;
	m_dbKohonenLearningRate     = 0.2;
	m_dbKohonenDecayRate        = 2.0;
	m_dbTolerance               = .2;
	m_dbGrossbergDecayRate      = 0.1;
	m_dbKohonenInitRange        = 1.0;
	m_dbInhibitingFactor        = 0.9;
	m_dbNoiseFactor             = .001;
	m_dwInputSize               = INPUT;
	m_dwOutputSize              = OUTPUT;
	m_dwHiddenSize              = HIDDEN;
	m_dwNumberOfWinnersAllowed  = 1;
	m_dwNumberOfIterations      = 1;

	DWORD i = 0, j = 0;

	unsigned int init = static_cast<unsigned int>( time(NULL) );
	srand( init );

	m_vvdKohonenMatrix.resize(m_dwHiddenSize);
	
	double random_value = CounterpropNetwork::random();

	for( i = 0; i < m_dwHiddenSize; i++ )
	{
		for ( j = 0; j < m_dwInputSize; j++ )
		{
			m_vvdKohonenMatrix[i].push_back( random_value );
		}
	}

	// initialize hidden neuron "wins" counters
	for( i = 0; i < m_dwHiddenSize; i++ )
	{
		m_viNumberOfWins.push_back( 0 );
	}

	// initialize Grossberg Matrix weights to 0
	m_vvdGrossbergMatrix.resize( m_dwOutputSize );

	for( i = 0; i < m_dwOutputSize; i++ )
	{
		for( j = 0; j < m_dwHiddenSize; j++ )
		{
			m_vvdGrossbergMatrix[i].push_back( 0.0 );
		}
	} 
};

CounterpropNetwork::~CounterpropNetwork()
{
	
}

// Training input vectors until all input vectors have been learned
void CounterpropNetwork::training( vector<vector<double>> vvd_trainingInputVector, vector<vector<double>> vvd_trainingOutputVector )
{
	DWORD dw_winningHiddenIndex,
		  dwTrainingInputVectorSize;
	unsigned int i;
	vector<double> res;

	int x        = 0;
	int epochs   = EPOCHS;
	double error = 0.0;

	bool bFirst = TRUE;

	dwTrainingInputVectorSize = vvd_trainingInputVector.size();

	while( epochs && ( error > PRECISION || bFirst ) )
	{
		error  = 0.0;
		bFirst = FALSE;

		for( i = 0; i < dwTrainingInputVectorSize; i++ )
		{
			dw_winningHiddenIndex = trainingVector( vvd_trainingInputVector[i] );

			m_viNumberOfWins[dw_winningHiddenIndex]++;

			updateGrossbergWeights( dw_winningHiddenIndex, 0, vvd_trainingOutputVector[i] );

			res = testing(vvd_trainingInputVector[i]);

			for( x = 0; x < OUTPUT; ++x )
			{
				error += pow( res[x] - vvd_trainingOutputVector[i][x], 2 );
			}
		}

		error /= OUTPUT;

		epochs--;
	}


}

// training of a single vector
int CounterpropNetwork::trainingVector( vector<double>& inputVector )
{
	// Normalize the input vector x
	//normalize( inputVector );

	// Determine winning node in the Kohonen layer
	int winningHiddenIndex = getWinningHiddenIndex( inputVector );

	// Update winning node's weight vector
	updateKohonenWeights( winningHiddenIndex, inputVector );

	return winningHiddenIndex;
}

vector<double> CounterpropNetwork::testing( vector<double>& inputVector )
{
	DWORD i, j;
		
	vector<double> outputVector( m_dwOutputSize );

	// present input vector and calculate hidden layer activations
	vector<double> hiddenActivations( m_dwHiddenSize );

	for( i = 0; i < m_dwHiddenSize; i++ )
	{
		for( j = 0; j < m_dwInputSize; j++ )
		{
			hiddenActivations[i] += m_vvdKohonenMatrix[i][j] * inputVector[j];
		}
	}

	// determine the winning hidden layer neuron
	// (the one with the maximum activation)
	int winningIndex = 0;
	double maxActivation = hiddenActivations[0];

	for( i = 0; i < m_dwHiddenSize; i++ )
	{
		if(hiddenActivations[i] > maxActivation )
		{
			maxActivation = hiddenActivations[i];
			winningIndex = i;
		}
	}

	// calculate the output vector
	// (the synaptic connections from the winning hidden neuron
	// to the output layer)
	for( i = 0; i < m_dwOutputSize; i++ )
	{
		outputVector[i] = m_vvdGrossbergMatrix[i][winningIndex];
	}

	return outputVector;
}

void CounterpropNetwork::printMatrix()
{
	DWORD i, j;
	// print Kohonen weights
	printf("Kohonen matrix:");
	for( i = 0; i < m_dwHiddenSize; i++ )
	{
		for( j = 0; j < m_dwInputSize; j++ )
			printf( "%f ", m_vvdKohonenMatrix[i][j] );

		printf("\n");
	}

	// print Grossberg weights
	for( i = 0; i < m_dwOutputSize; i++ )
	{
		for( j = 0; j < m_dwHiddenSize; j++ )
		{
			printf( "%f ", m_vvdGrossbergMatrix[i][j] );
		}
		printf("\n");
	}         
}


void CounterpropNetwork::updateKohonenWeights( int hiddenIndex, vector<double>& inputVector )
{
	DWORD i;

	for( i = 0; i < m_dwInputSize; i++ )
	{       
		m_vvdKohonenMatrix[hiddenIndex][i] += m_dbKohonenLearningRate * (inputVector[i] - m_vvdKohonenMatrix[hiddenIndex][i]);
	}
}


void CounterpropNetwork::updateGrossbergWeights( DWORD hiddenIndex, double hiddenActivation, vector<double>& outputVector )
{
	DWORD i;

	for( i = 0; i < m_dwOutputSize; i++ )
	{
		m_vvdGrossbergMatrix[i][hiddenIndex] += m_dbGrossbergLearningRate * ( outputVector[i] - m_vvdGrossbergMatrix[i][hiddenIndex] );
	}
}

int CounterpropNetwork::getWinningHiddenIndex(vector<double>& vd_inputVector)
{
	int theWinningIndex = -1;
	double minimumDistance = DBL_MAX;
	double distance;
	DWORD i, j;

	for( i = 0; i < m_dwHiddenSize; i++ )
	{

		// compute the distance between the synaptic weights and
		// the input vector
		distance = 0.0;

		for( j = 0; j < m_dwInputSize; j++)
		{
			distance += pow( ( m_vvdKohonenMatrix[i][j] - vd_inputVector[j] ), 2 );
		}

		distance = pow(distance, .5);

		// scale this distance to inhibit the neuron from winning again
		distance *= fabs( 1 + ( (double)m_viNumberOfWins[i] * m_dbInhibitingFactor ) );

		if( distance < minimumDistance )
		{
			minimumDistance = distance;
			theWinningIndex = i;
		}
	}

	return theWinningIndex;
}

void CounterpropNetwork::normalize( vector<double>& vd_inputVector )
{
	unsigned int i;
	double sum = 0;
	double norm;

	for( i = 0; i < vd_inputVector.size(); i++ )
		sum += pow( vd_inputVector[i], 2 );

	norm = pow( sum, 0.5 );

	for( i = 0; i < vd_inputVector.size(); i++ )
		vd_inputVector[i] /= norm;
}