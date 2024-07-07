#include "defs.h"
#include <stdlib.h>
#include <vector>
#include <immintrin.h>

struct LorenzoConfig {
        double error;
        double first;
        double outiers[0];
};

static inline int32_t diff(const double &data, const double &predicted, const DOUBLE &e, const double &e2, const double &max_diff) {
        DOUBLE d = {.d = data - predicted};
        DOUBLE d_abs = {.i = d.i & ~DOUBLE_SIGN_BIT};
        if (UNLIKELY(d_abs.d > max_diff)) {
                return INT32_MIN;
        }
        DOUBLE comp = {.i = (d.i & DOUBLE_SIGN_BIT) | e.i};
        d.d += comp.d;
        return static_cast<int32_t>(d.d / e2);
}

ssize_t lorenzo1_diff(double* input, ssize_t len, int32_t* output, double error, uint8_t** predictor_out, ssize_t* psize) {
        std::vector<double> outier;
        DOUBLE e = {.d= error * 0.999};
        double e2 = e.d * 2;
        double max_diff = e2 * INT32_MAX;
        double predicted = input[0];
        for (int i = 1; i < len; i++) {
                *output = diff(input[i], predicted, e, e2, max_diff);
                if (UNLIKELY(*output == INT32_MIN)) {
                        outier.push_back(input[i]);
                        predicted = input[i];
                } else {
                        predicted += *output * e2;
                }
                output++;
        }
        *psize = sizeof(LorenzoConfig) + outier.size() * sizeof(double);
        *predictor_out = reinterpret_cast<uint8_t*>(malloc(*psize));
        LorenzoConfig* config = reinterpret_cast<LorenzoConfig*>(*predictor_out);
        config->error = error;
        config->first = input[0];
        __builtin_memcpy(config->outiers, &outier[0], outier.size() * sizeof(double));
        return len - 1;
}

ssize_t lorenzo1_correct(int32_t* input, ssize_t len, double* output, uint8_t* predictor_out, ssize_t psize) {
        LorenzoConfig* config = reinterpret_cast<LorenzoConfig*>(predictor_out);
        double e2 = config->error * 0.999 * 2;
        
        output[0] = config->first; 
        if (psize == sizeof(LorenzoConfig)) {
                for (int i = 0; i < len; i++) {
                        output[i+1] = output[i] + e2 * input[i];
                }
        } else {
                double* outier = config->outiers;
                for (int i = 0; i < len; i++) {
                        if (UNLIKELY(input[i] == INT32_MIN)) {
                                output[i+1] = *outier++;
                        } else {
                                output[i+1] = output[i] + e2 * input[i];
                        }
                }
        }
        return len + 1;
}


ssize_t simd_lorenzo1_diff(double* input, ssize_t len, int32_t* output, double error, uint8_t** predictor_out, ssize_t* psize) {
        std::vector<double> outier;
        DOUBLE e = {.d= error * 0.999};
        double e2 = e.d * 2;
        double max_diff = e2 * INT32_MAX;
        double predicted = input[0];
        for (int i = 1; i < len; i++) {
                *output = diff(input[i], predicted, e, e2, max_diff);
                if (UNLIKELY(*output == INT32_MIN)) {
                        outier.push_back(input[i]);
                        predicted = input[i];
                } else {
                        predicted += *output * e2;
                }
                output++;
        }
        *psize = sizeof(LorenzoConfig) + outier.size() * sizeof(double);
        *predictor_out = reinterpret_cast<uint8_t*>(malloc(*psize));
        LorenzoConfig* config = reinterpret_cast<LorenzoConfig*>(*predictor_out);
        config->error = error;
        config->first = input[0];
        __builtin_memcpy(config->outiers, &outier[0], outier.size() * sizeof(double));
        return len - 1;
}

ssize_t simd_lorenzo1_correct(int32_t* input, ssize_t len, double* output, uint8_t* predictor_out, ssize_t psize) {
        LorenzoConfig* config = reinterpret_cast<LorenzoConfig*>(predictor_out);
        double e2 = config->error * 0.999 * 2;
        
        output[0] = config->first; 
        
        if (psize == sizeof(LorenzoConfig)) {
                //Handle the unaligned 32 Bytes.
                int i = 0;
                while(reinterpret_cast<uintptr_t>(&input[i]) % 32 !=0){
                        output[i+1] = output[i] + e2 * input[i];
                        ++i;
                }
                int32_t temp_array[8];
                for (; i+8 < len; i+=8) {
                        /**
                         * Step 1. Calculate Presum.
                        */
                        __m256i x = _mm256_load_si256((__m256i*)(&(input[i])));  
                        // x = (1, 2, 3, 4, 5, 6, 7, 8)              
                        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
                        // x = (1, 1+2, 2+3, 3+4, 5, 5+6, 6+7, 7+8)
                        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
                        // x = (1, 1+2, 1+2+3, 1+2+3+4, 5, 5+6, 5+6+7, 5+6+7+8)
                        int32_t fourth_value = _mm256_extract_epi32(x, 3);
                        // extract fourth value(1+2+3+4)
                        __m256i add_value = _mm256_set1_epi32(fourth_value);
                        // add_value all eqauls fourth value
                        __m256i sum_value = _mm256_add_epi32(x, add_value);
                        // sum_value = (1+1+2+3+4, 1+2+1+2+3+4, 1+2+3+1+2+3+4, 1+2+3+4+1+2+3+4, 5+1+2+3+4, 5+6+1+2+3+4, 5+6+7+1+2+3+4, 5+6+7+8+1+2+3+4)
                        x = _mm256_blend_epi32(x, sum_value, 0xF0);
                        //Only remain high 4 int32_t, x = (1, 1+2, 1+2+3, 1+2+3+4, 1+2+3+4+5, 1+2+3+4+5+6, 1+2+3+4+5+6+7, 1+2+3+4+5+6+7+8)

                        /**
                         * Step 2. ALU Calculation 
                        */
                        //Convert to double
                        __m512i x_512 = _mm512_cvtepi32_epi64(x);
                        __m512d x_d = _mm512_cvtepi64_pd(x_512);
                        __m512d e_vec = _mm512_set1_pd(e2);
                        __m512d out_vec = _mm512_set1_pd(output[i]);
                        //Calculate output = out_vec +  e_vec * x
                        __m512d result =  _mm512_add_pd(_mm512_mul_pd(x_d, e_vec), out_vec);
                        //Store result
                        _mm512_storeu_pd(output+i+1, result);
                }
                //Handle the left Bytes.
                for (; i < len; i++) {
                        output[i+1] = output[i] + e2 * input[i];
                }
        } else {
                double* outier = config->outiers;
                for (int i = 0; i < len; i++) {
                        if (UNLIKELY(input[i] == INT32_MIN)) {
                                output[i+1] = *outier++;
                        } else {
                                output[i+1] = output[i] + e2 * input[i];
                        }
                }
        }
        return len + 1;
}



