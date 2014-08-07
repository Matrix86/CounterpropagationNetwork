#include "counterpropagation.h"

double random()
{
	double random_value;

	random_value = ( (double)( ( rand() % 100 ) + 1 ) ) / 101.0;

	return random_value;
}

void main(int argc, char *argv[])
{
	// --- define training and testing sets ---
	unsigned int init = static_cast<unsigned int> ( time(NULL) );
	srand( init );

	int i, pair;
	char str[8] = {0};

	// number of input-output vector pairs in the training set
	int numberOfTrainingPairs = HIDDEN;

	// number of input-output vector pairs in the testing set
	int numberOfTestingPairs  = 6;

	// training set
	vector<vector<double>> trainingInputVector;
	vector<vector<double>> trainingOutputVector;

	// testing set
	vector<vector<double>> testingInputVector;
	vector<vector<double>> testingOutputVector;

	// --- read in the training set ---
	try
	{ 
		ifstream hfile( "train.txt" );

		if( !hfile.good() ) 
		{
			printf( "File not exists\n" );

			return;
		}

		// read each line of the file
		// each line contains an input-output vector pair of the form:
		//
		// input_0 input_1 ... input_n output_0 output_1 ... output_n
		trainingInputVector.resize( numberOfTrainingPairs );
		trainingOutputVector.resize( numberOfTrainingPairs );

		for( pair = 0; pair < numberOfTrainingPairs; pair++)
		{
			string line;


			getline( hfile, line );

			istringstream split(line);
			string token;

			// read in the input vector
			for( i = 0; i < INPUT; i++ )
			{
				getline( split, token, ' ' );
				trainingInputVector[pair].push_back( atof( token.c_str() ) );
			}

			// read in the output vector
			for( i = 0; i < OUTPUT; i++ )
			{
				getline( split, token, ' ' );
				trainingOutputVector[pair].push_back( atof( token.c_str() ) );
			}
		} 
	}
	catch (exception e) 
	{ 
		printf( e.what() ); 
	}

	// --- read in the testing set ---

	try
	{
		ifstream hfile( "test.txt" );
		if( !hfile.good() ) 
		{
			printf( "File not exists\n" );
			return;
		}

		// read each line of the file
		// each line contains an input-output vector pair of the form:
		//
		// input_0 input_1 ... input_n output_0 output_1 ... output_n
		testingInputVector.resize( numberOfTestingPairs );
		testingOutputVector.resize( numberOfTestingPairs );

		for( pair = 0; pair < numberOfTestingPairs; pair++ )
		{
			string line;
			getline( hfile, line );

			istringstream iss(line);
			string token;

			// read in the input vector
			for( int i = 0; i < INPUT; i++ )
			{
				getline(iss, token, ' ');
				testingInputVector[pair].push_back( atof(token.c_str() ) );
			}

			// read in the output vector
			for( int i = 0; i < OUTPUT; i++ )
			{
				getline( iss, token, ' ' );
				testingOutputVector[pair].push_back( atof( token.c_str() ) );
			}


		} 
	}
	catch ( exception e ) 
	{ 
		printf( e.what() ); 
	}

	// network init

	CounterpropNetwork *c = new CounterpropNetwork();

	// train the network
	c->training( trainingInputVector, trainingOutputVector );

	printf("Testing network...\n");

	// test the network

	double error = 0;

	vector<double> v_error(OUTPUT);
	vector<int>    v_numCorrect(OUTPUT);

	for( i = 0; i < numberOfTestingPairs; i++ )
	{

		vector<double> outputVector( OUTPUT );
		
		outputVector = c->testing( testingInputVector[i] );

		printf( "Output for line #%d: ", i+1 );

		for( int x = 0; x < OUTPUT; ++x )
		{
			printf( "%.1f ", outputVector[x] );
		}

		printf( "(Real: " );

		for( int x = 0; x < OUTPUT; ++x )
		{
			printf( "%.1f ", testingOutputVector[i][x] );
		}

		printf( ")" );

		printf( "\n" );

		for( int x = 0; x < OUTPUT; ++x )
		{
			error = fabs( outputVector[x] - testingOutputVector[i][x] );

			if( error <= c->m_dbTolerance )
				v_numCorrect[x]++;
		}
	}

	// output testing accuracy
	for( int x = 0; x < OUTPUT; ++x )
	{
		v_error[x] = (double) v_numCorrect[x] / numberOfTestingPairs * 100.00;

		printf( "Accuracy Output %d = %.0f%%\n", x, v_error[x] );
	}
}