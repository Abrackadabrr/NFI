#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>

const int NUM_THREADS = 4;
const int ITERATIONS = 1000000000;  // 100 млн операций
const int CACHE_LINE_SIZE = 64;

// Вариант 1: Счетчики рядом (false sharing)
struct FalseSharing {
    int counter1;
    int counter2;
    int counter3;
    int counter4;
};

// Вариант 2: Счетчики с выравниванием (нет false sharing)
struct alignas(CACHE_LINE_SIZE) NoFalseSharing {
    volatile int counter;
};

int main() {
    std::cout << "=== ЭКСПЕРИМЕНТ: FALSE SHARING ===\n";
    std::cout << "Каждый поток делает " << ITERATIONS << " инкрементов\n\n";
    
    // ЭКСПЕРИМЕНТ 1: FALSE SHARING - ВСЕ СЧЕТЧИКИ РЯДОМ
    {
        FalseSharing data = {0, 0, 0, 0};
        
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            int tid = omp_get_thread_num();
            
            // КРИТИЧНО: каждый поток постоянно пишет в свой счетчик
            for (int i = 0; i < ITERATIONS; i++) {
                if (tid == 0) data.counter1++;
                else if (tid == 1) data.counter2++;
                else if (tid == 2) data.counter3++;
                else if (tid == 3) data.counter4++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "1. FALSE SHARING (счетчики рядом): " << ms.count() << " мс\n";
        std::cout << "   Значения: " << data.counter1 << " " << data.counter2 << " "
                  << data.counter3 << " " << data.counter4 << "\n";
    }
    
    // ЭКСПЕРИМЕНТ 2: NO FALSE SHARING - КАЖДЫЙ СЧЕТЧИК НА СВОЕЙ КЭШ-ЛИНИИ
    {
        // Выделяем память с выравниванием
        NoFalseSharing* data = (NoFalseSharing*)aligned_alloc(
            CACHE_LINE_SIZE, NUM_THREADS * sizeof(NoFalseSharing)
        );
        for (int i = 0; i < NUM_THREADS; i++) data[i].counter = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            int tid = omp_get_thread_num();
            
            // Каждый поток работает со СВОИМ счетчиком на отдельной кэш-линии
            for (int i = 0; i < ITERATIONS; i++) {
                data[tid].counter++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "2. NO FALSE SHARING (выровнено): " << ms.count() << " мс\n";
        std::cout << "   Значения: " << data[0].counter << " " << data[1].counter << " "
                  << data[2].counter << " " << data[3].counter << "\n";
        
        free(data);
    }
     
    return 0;
}
