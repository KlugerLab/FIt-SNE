#ifndef TIME_CODE_H
#define TIME_CODE_H
#include <chrono>
#if defined(TIME_CODE)
#pragma message "Timing code"                                   
#define INITIALIZE_TIME std::chrono::steady_clock::time_point STARTVAR;                       
#define START_TIME           			\
                STARTVAR = std::chrono::steady_clock::now();
                                                               
#define END_TIME(LABEL) {          			\
                std::chrono::steady_clock::time_point ENDVAR = std::chrono::steady_clock::now();      \
                printf("%s: %ld ms\n",LABEL, std::chrono::duration_cast<std::chrono::milliseconds>(ENDVAR-STARTVAR).count());       \
}                                                                       
#else 
#define INITIALIZE_TIME                      
#define START_TIME           			
#define END_TIME(LABEL) {}                                                                       

#endif
#endif
