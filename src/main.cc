#include <iostream>
#include <cstdint>
using namespace std;

#include <unistd.h>
#include <cmath>

#include "AudioFile.h"

#define MAX_LPC_ORDER 17  // Maximum LPC order

#include <liquid/liquid.h>

#include <dlib/svm.h>

using namespace std;
using namespace dlib;

int fast_decode(unsigned int x) {
    return (x >> 1) ^ (-(x&1));
}

unsigned int fast_encode(int x) {
    return (2*x) ^ (x >>(sizeof(int) * 8 - 1));
}

// Function to pack bits from a vector of any unsigned integer type
template <typename T>
std::vector<T> packBits(const std::vector<T>& arr) {
    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");

    const size_t bitsPerWord = sizeof(T) * 8;  // Number of bits in the integer type (e.g., 16, 32, 64)
    size_t numElements = arr.size();

    // The total number of words needed to store all the packed bits.
    size_t numWords = (numElements * bitsPerWord + bitsPerWord - 1) / bitsPerWord;
    
    std::vector<T> packedBits(numWords, 0);  // Initialize all words to 0

    // Loop through each bit position (0 to bitsPerWord - 1)
    for (size_t bitPos = 0; bitPos < bitsPerWord; ++bitPos) {
        for (size_t i = 0; i < numElements; ++i) {
            // Extract the bit at bitPos from the current element
            T bit = (arr[i] >> bitPos) & 1;

            // Find where to store this bit in the packedBits vector
            size_t bitIndex = bitPos * numElements + i;
            size_t wordIndex = bitIndex / bitsPerWord;
            size_t bitOffset = bitIndex % bitsPerWord;

            // Set the appropriate bit in the packedBits word
            packedBits[wordIndex] |= (static_cast<T>(bit) << bitOffset);
        }
    }

    return packedBits;
}

// Function to unpack bits from a vector of any unsigned integer type
template <typename T>
std::vector<T> unpackBits(const std::vector<T>& packedBits, size_t numElements) {
    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");

    const size_t bitsPerWord = sizeof(T) * 8;  // Number of bits in the integer type
    std::vector<T> unpackedArray(numElements, 0);  // Initialize the result array with zeros

    // Loop through each bit position (0 to bitsPerWord - 1)
    for (size_t bitPos = 0; bitPos < bitsPerWord; ++bitPos) {
        for (size_t i = 0; i < numElements; ++i) {
            // Calculate the index and offset for where this bit is stored in packedBits
            size_t bitIndex = bitPos * numElements + i;
            size_t wordIndex = bitIndex / bitsPerWord;
            size_t bitOffset = bitIndex % bitsPerWord;

            // Extract the bit from the packedBits
            T bit = (packedBits[wordIndex] >> bitOffset) & 1;

            // Place the extracted bit back into the appropriate position in the unpacked array
            unpackedArray[i] |= (bit << bitPos);
        }
    }

    return unpackedArray;
}

template <typename T>
double computeRatioNonNull(const std::vector<T>& vec) {
    if (vec.empty()) {
        // Handle edge case of empty vector to avoid division by zero
        std::cerr << "The vector is empty!" << std::endl;
        return 0.0;
    }

    int nonNullCount = 0;
    for (const auto& elem : vec) {
        if (elem != 0.0) {
            ++nonNullCount;
        }
    }

    // Calculate ratio of non-null elements to total elements
    return static_cast<double>(nonNullCount) / vec.size();
}

template <typename T>
void savePackedData(const std::string& outputFilePath, std::vector<T>& vec) {
    if (vec.empty()) {
        // Handle edge case of empty vector to avoid division by zero
        std::cerr << "The vector is empty!" << std::endl;
    }
    ofstream ofs(outputFilePath);
    ofs.write(reinterpret_cast<char*>(&vec[0]), sizeof(vec[0]) * vec.size());
    ofs.close();
}

// Function to compute autocorrelation
template <typename T>
std::vector<double> compute_autocorrelation(const std::vector<T>& signal, int order) {
    int N = signal.size();
    std::vector<double> autocorr(order + 1, 0.0);

    // Compute the autocorrelation R[k]
    for (int k = 0; k <= order; ++k) {
        for (int n = 0; n < N - k; ++n) {
            autocorr[k] += (double)signal[n] * (double)signal[n + k];
        }
    }

    return autocorr;
}


// Levinson-Durbin Algorithm for LPC Coefficients
void levinson_durbin(const int n, const std::vector<double>& r, std::vector<double>& a, std::vector<double>& k) {
    std::vector<double> a_temp(MAX_LPC_ORDER, 0.0);  // Temporary array for LPC coefficients
    double alpha, epsilon;
    int i, j;

    k[0] = 0.0;  // k[0] is unused
    a[0] = 1.0;  // a[0] is always 1.0
    a_temp[0] = 1.0;  // Unnecessary but consistent

    alpha = r[0];

    for (i = 1; i <= n; ++i) {
        epsilon = r[i];
        for (j = 1; j < i; ++j) {
            epsilon += a[j] * r[i - j];
        }

        k[i] = -epsilon / alpha;
        alpha = alpha * (1.0 - k[i] * k[i]);

        for (j = 1; j < i; ++j) {
            a_temp[j] = a[j] + k[i] * a[i - j];
        }
        for (j = 1; j < i; ++j) {
            a[j] = a_temp[j];
        }
        a[i] = k[i];
    }
}

// Function to predict values using LPC coefficients
template <typename T>
std::vector<T> predict_signal(const std::vector<T>& signal, const std::vector<double>& lpc_coeffs, int order) {
    float a_hat[order+1];   // lpc output
    float g_hat[order+1];   // lpc output
    for(int i = 0; i < order+1; ++i ){
        a_hat[i] = lpc_coeffs[i];
        g_hat[i] = lpc_coeffs[i+order+1];
    }
    // run prediction filter
    float a_lpc[order+1];
    float b_lpc[order+1];
    for (int i=0; i<order+1; i++) {
        a_lpc[i] = (i==0) ? 1.0f : 0.0f;
        b_lpc[i] = (i==0) ? 0.0f : -a_hat[i];
    }
    iirfilt_rrrf f = iirfilt_rrrf_create(b_lpc,order+1, a_lpc,order+1);
    iirfilt_rrrf_print(f);
    float y_hat[signal.size()];
    for (int i=0; i<signal.size(); i++)
        iirfilt_rrrf_execute(f, (float)(signal[i]), &y_hat[i]);
    iirfilt_rrrf_destroy(f);
    std::vector<T> ret(signal.size());
    for(int i = 0; i < signal.size(); ++i )
    {
        ret[i] = y_hat[i];
    }
    return ret;
}

// Function to compute LPC coefficients
template <typename T>
std::vector<double> compute_lpc(const std::vector<T>& signal, int order) {
    std::vector<double> ret;
    float a_hat[order+1];   // lpc output
    float g_hat[order+1];   // lpc output
    std::vector<float> input;
    for( auto& s : signal )
    {
        input.push_back(s);
    }
    liquid_lpc(&input[0],signal.size(),order,a_hat,g_hat);
    for(size_t i = 0; i < order+1; ++i)
    {
        ret.push_back(a_hat[i]);
    }
    for(size_t i = 0; i < order+1; ++i)
    {
        ret.push_back(g_hat[i]);
    }
    return ret;
}

typedef int32_t sampleType;

int main(int argc, char** argv) 
{

    // options
    unsigned int n = 200;   // input sequence length
    unsigned int p = 4;     // prediction filter order

    // create low-pass filter object
    iirfilt_rrrf f = iirfilt_rrrf_create_lowpass(2, 0.05f);
    iirfilt_rrrf_print(f);

    unsigned int i;

    // allocate memory for data arrays
    float y[n];         // input signal (filtered noise)
    float a_hat[p+1];   // lpc output
    float g_hat[p+1];   // lpc output

    // generate input signal (filtered noise)
    for (i=0; i<n; i++)
        iirfilt_rrrf_execute(f, randnf(), &y[i]);

    // destroy filter object
    iirfilt_rrrf_destroy(f);

    // run linear prediction algorithm
    liquid_lpc(y,n,p,a_hat,g_hat);

    // run prediction filter
    float a_lpc[p+1];
    float b_lpc[p+1];
    for (i=0; i<p+1; i++) {
        a_lpc[i] = (i==0) ? 1.0f : 0.0f;
        b_lpc[i] = (i==0) ? 0.0f : -a_hat[i];
    }
    f = iirfilt_rrrf_create(b_lpc,p+1, a_lpc,p+1);
    iirfilt_rrrf_print(f);
    float y_hat[n];
    for (i=0; i<n; i++)
        iirfilt_rrrf_execute(f, y[i], &y_hat[i]);
    iirfilt_rrrf_destroy(f);

    // compute prediction error
    float err[n];
    for (i=0; i<n; i++)
        err[i] = y[i] - y_hat[i];

    // compute autocorrelation of prediction error
    float lag[n];
    float rxx[n];
    for (i=0; i<n; i++) {
        lag[i] = (float)i;
        rxx[i] = 0.0f;
        unsigned int j;
        for (j=i; j<n; j++)
            rxx[i] += err[j] * err[j-i];
    }
    float rxx0 = rxx[0];
    for (i=0; i<n; i++)
        rxx[i] /= rxx0;

    // print results
    for (i=0; i<p+1; i++)
        printf("  a[%3u] = %12.8f, g[%3u] = %12.8f\n", i, a_hat[i], i, g_hat[i]);

    printf("  prediction rmse = %12.8f\n", sqrtf(rxx0 / n));

    // return 0;

    std::string inputPath = argv[1];

    AudioFile<sampleType> audioFile;
    audioFile.load (inputPath);

    int sampleRate = audioFile.getSampleRate();
    int bitDepth = audioFile.getBitDepth();

    int numSamples = audioFile.getNumSamplesPerChannel();
    double lengthInSeconds = audioFile.getLengthInSeconds();

    int numChannels = audioFile.getNumChannels();
    bool isMono = audioFile.isMono();
    bool isStereo = audioFile.isStereo();

    // or, just use this quick shortcut to print a summary to the console
    audioFile.printSummary();
    // return 0;

    size_t blockSize = 12500*2;
    std::vector<sampleType> sig_block;
    std::vector<uint32_t> block;
    std::vector<uint32_t> block2;
    int channel = 0;
    sampleType t_n_1;
    sampleType t_n_2;
    for (int i = 0; i < numSamples; i++)
    {
        sampleType t_n = audioFile.samples[channel][i];
        if( i > 1 ) {
            // sampleType D = fast_encode(( t_n - t_n_1 ) - ( t_n_1 - t_n_2 ));
            // cerr << t_n << " ~ " << D << endl;
            sig_block.push_back(t_n);
            // block.push_back( d );
            if( sig_block.size() >= blockSize ) {

                const size_t maxWindow = 4;
                typedef matrix<double,maxWindow,1> sample_type;

                // Learning
                std::deque<double> dodWindow;
                // Here we declare that our samples will be 1 dimensional column vectors.  
                typedef matrix<double,maxWindow,1> sample_type;
                sample_type m;
                std::vector<sample_type> samples;
                std::vector<double> labels;
                uint16_t lastSample = 0;
                for (int i = 0; i < sig_block.size(); i++)
                {
                    double isample = sig_block[i];
                    if (dodWindow.size() >= maxWindow)
                    {
                        for( int k = 0; k < maxWindow; ++k )
                        {
                            m(k) = dodWindow[k];
                        }
                        samples.push_back(m);
                        labels.push_back(isample);
                    }
                    dodWindow.push_back(isample);
                    while( dodWindow.size() > maxWindow )
                        dodWindow.pop_front();
                }
                dodWindow.clear();
                // Now we are making a typedef for the kind of kernel we want to use.  I picked the
                // radial basis kernel because it only has one parameter and generally gives good
                // results without much fiddling.
                typedef radial_basis_kernel<sample_type> kernel_type;

                // Here we declare an instance of the krr_trainer object.  This is the
                // object that we will later use to do the training.
                krr_trainer<kernel_type> trainer;

                // Here we set the kernel we want to use for training.   The radial_basis_kernel 
                // has a parameter called gamma that we need to determine.  As a rule of thumb, a good 
                // gamma to try is 1.0/(mean squared distance between your sample points).  So 
                // below we are using a similar value computed from at most 2000 randomly selected
                // samples.
                const double gamma = 1.0/compute_mean_squared_distance(randomly_subsample(samples, 2000));
                cout << "using gamma of " << gamma << endl;
                trainer.set_kernel(kernel_type(gamma));

                // now train a function based on our sample points
                decision_function<kernel_type> test = trainer.train(samples, labels);
                //
                //

                int lpc_order = 3;
                auto coeffs = compute_lpc(sig_block, lpc_order);
                for( auto lpcv : coeffs ) {
                    cerr << "LPC: " << lpcv << endl;
                }

                // Predict the signal values using the LPC coefficients
                std::vector<sampleType> predicted_signal = predict_signal(sig_block, coeffs, lpc_order);
                // Print the original vs predicted signal values
                // std::cout << "Original vs Predicted Signal Values:" << std::endl;
                for (int i = 0; i < sig_block.size(); ++i) {
                    // KRLS
                    double isample = sig_block[i];
                    if (dodWindow.size() >= maxWindow)
                    {
                        for( int k = 0; k < maxWindow; ++k )
                        {
                            m(k) = dodWindow[k];
                        }
                        auto prediction = test(m);
                        // std::cout << "[KRLS] Original: " << sig_block[i+1] << ", Predicted: " << prediction << std::endl;
                        int32_t randError = -2 + ::rand() % 4;
                        sampleType d2 = fast_encode( prediction - sig_block[i] );
                        // sampleType d2 = fast_encode( randError );
                        block2.push_back(d2);
                    }
                    dodWindow.push_back(isample);
                    while( dodWindow.size() > maxWindow )
                        dodWindow.pop_front();
                    //
                    // std::cout << "Original: " << sig_block[i] << ", Predicted: " << predicted_signal[i] << std::endl;
                    sampleType d2 = fast_encode( predicted_signal[i] - sig_block[i] );
                    sampleType d = fast_encode( ( sig_block[i] - sig_block[i-1] ) );
                    // cerr << "d=" << d << endl;
                    block.push_back(d2);
                    // block2.push_back(d2);
                }

                auto papack = packBits(block);
                auto papack2 = packBits(block2);

                // auto unpapack = unpackBits(papack, papack.size());
                // for( auto v : papack ) {
                //     cerr << "* " << v << endl;
                // }
                
                cerr << "SIG_BLOCK.size()=" << sig_block.size() << " PACKED.size()=" << papack.size() << endl;
                cerr << "NAIVE: " << computeRatioNonNull(papack) << endl;
                cerr << "LPC: " << computeRatioNonNull(papack2) << endl;

                savePackedData( "/tmp/orig.bin", sig_block );
                savePackedData( "/tmp/data.bin", papack2 );

                sleep(1);
                block.clear();
                block2.clear();
                sig_block.clear();
            }
        }
        t_n_2 = t_n_1;
        t_n_1 = t_n;
    }
    cerr << endl;

    return 0;
}