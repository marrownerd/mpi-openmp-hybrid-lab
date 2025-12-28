#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <iostream>
#include <queue>
#include <vector>
#include <pthread.h>
#include <unistd.h>

struct Job {
int jobSimulationTime;
Job(int jst) : jobSimulationTime(jst) {}
Job() : jobSimulationTime(-1) {}
code Code

    
void process() {
    sleep(jobSimulationTime);
}

  

};

// Глобальные переменные
std::queue<Job> jobQueue;
bool canBeMoreWork = true;
bool allWorkFinished = false;

// Примитивы синхронизации
pthread_mutex_t mutex;
pthread_cond_t c_jobAdded, c_needToFetch;
bool needToFetch = false;
int fetchThreshold = 3; // Порог для запроса новых задач

// Worker-поток - обрабатывает задачи
void* worker(void* args) {
int localJobsProcessed = 0;
code Code

    
while (true) {
    Job job;
    bool gotJob = false;
    
    pthread_mutex_lock(&mutex);
    
    // Если задач мало и мы еще можем получить больше - сигнализируем fetcher
    if (jobQueue.size() <= fetchThreshold && !needToFetch && canBeMoreWork) {
        needToFetch = true;
        pthread_cond_signal(&c_needToFetch);
    }
    
    // Ждем задачи, если очередь пуста, но работа еще может быть
    while (jobQueue.empty() && canBeMoreWork && !allWorkFinished) {
        pthread_cond_wait(&c_jobAdded, &mutex);
    }
    
    // Берем задачу из очереди
    if (!jobQueue.empty()) {
        job = jobQueue.front();
        jobQueue.pop();
        gotJob = true;
        localJobsProcessed++;
    }
    
    pthread_mutex_unlock(&mutex);
    
    // Если задача получена - обрабатываем
    if (gotJob) {
        std::cout << "Process " << ((int*)args)[1] << ": Processing job for " 
                  << job.jobSimulationTime << " seconds" << std::endl;
        job.process();
    } 
    // Если работы нет и больше не будет - выходим
    else if (allWorkFinished) {
        std::cout << "Process " << ((int*)args)[1] << ": Worker finished. Processed " 
                  << localJobsProcessed << " jobs." << std::endl;
        break;
    }
}

return NULL;

  

}

// Fetcher-поток - запрашивает задачи у других процессов
void* fetcher(void* args) {
int group_size = ((int*)args)[0];
int my_rank = ((int*)args)[1];
std::vector<bool> finished(group_size, false);
finished[my_rank] = true; // Не запрашиваем у себя
code Code

    
while (canBeMoreWork) {
    pthread_mutex_lock(&mutex);
    
    // Ждем сигнала от worker, что нужно искать задачи
    while (!needToFetch && canBeMoreWork) {
        pthread_cond_wait(&c_needToFetch, &mutex);
    }
    
    // Сбрасываем флаг
    needToFetch = false;
    pthread_mutex_unlock(&mutex);
    
    // Если вся работа завершена - выходим
    if (!canBeMoreWork) break;
    
    // Запрашиваем задачи у других процессов
    bool fetchedJobs = false;
    for (int r = 0; r < group_size; r++) {
        if (!finished[r]) {
            int request = 1; // Запрос задачи
            int response = -1;
            
            // Отправляем запрос
            MPI_Send(&request, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            
            // Получаем ответ (длительность задачи или -1 если нет задач)
            MPI_Recv(&response, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (response > 0) {
                // Получили задачу
                pthread_mutex_lock(&mutex);
                jobQueue.push(Job(response));
                pthread_cond_signal(&c_jobAdded); // Будим worker
                pthread_mutex_unlock(&mutex);
                
                fetchedJobs = true;
                std::cout << "Process " << my_rank << ": Fetched job from process " 
                          << r << " with duration " << response << std::endl;
            } 
            else if (response == -1) {
                // У процесса нет задач
                finished[r] = true;
                
                // Проверяем, все ли процессы завершили работу
                bool allFinished = true;
                for (int i = 0; i < group_size; i++) {
                    if (!finished[i]) {
                        allFinished = false;
                        break;
                    }
                }
                
                if (allFinished) {
                    canBeMoreWork = false;
                    allWorkFinished = true;
                    pthread_mutex_lock(&mutex);
                    pthread_cond_broadcast(&c_jobAdded); // Будим всех workers
                    pthread_mutex_unlock(&mutex);
                }
            }
        }
    }
    
    // Если не получили задач, ждем немного перед следующей попыткой
    if (!fetchedJobs) {
        sleep(1);
    }
}

std::cout << "Process " << my_rank << ": Fetcher finished" << std::endl;
return NULL;

  

}

// Responder-поток - отвечает на запросы других процессов
void* responder(void* args) {
int group_size = ((int*)args)[0];
int my_rank = ((int*)args)[1];
int requestsHandled = 0;
code Code

    
while (canBeMoreWork) {
    int request = 0;
    MPI_Status status;
    
    // Неблокирующая проверка на запрос
    int flag = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
    
    if (flag) {
        // Получаем запрос
        MPI_Recv(&request, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        int response = -1;
        
        pthread_mutex_lock(&mutex);
        if (!jobQueue.empty() && jobQueue.size() > fetchThreshold) {
            // Отдаем задачу, если у нас достаточно задач
            Job job = jobQueue.front();
            jobQueue.pop();
            response = job.jobSimulationTime;
        }
        pthread_mutex_unlock(&mutex);
        
        // Отправляем ответ
        MPI_Send(&response, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        
        requestsHandled++;
        std::cout << "Process " << my_rank << ": Responded to process " 
                  << status.MPI_SOURCE << " with " << response << std::endl;
    } 
    else {
        // Ждем немного, если нет запросов
        usleep(100000); // 100ms
    }
}

std::cout << "Process " << my_rank << ": Responder finished. Handled " 
          << requestsHandled << " requests." << std::endl;
return NULL;

  

}

// Инициализация задач
void initializeJobs(int rank) {
pthread_mutex_lock(&mutex);
code Code

    
// Каждый процесс начинает с разным количеством задач
int initialJobs = 5 + (rank % 3); // 5-7 задач на процесс

for (int i = 0; i < initialJobs; i++) {
    jobQueue.push(Job(1 + (rand() % 3))); // Задачи на 1-3 секунды
}

std::cout << "Process " << rank << ": Initialized with " << initialJobs << " jobs" << std::endl;
pthread_mutex_unlock(&mutex);

  

}

int main(int argc, char** argv) {
int required = MPI_THREAD_MULTIPLE;
int provided = -1;
int size, rank;
code Code

    
MPI_Init_thread(&argc, &argv, required, &provided);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (required != provided) {
    std::cerr << "Your MPI implementation does not support MPI_THREAD_MULTIPLE." 
              << " Cannot proceed." << std::endl;
    abort();
}

// Инициализация примитивов синхронизации
pthread_mutex_init(&mutex, NULL);
pthread_cond_init(&c_jobAdded, NULL);
pthread_cond_init(&c_needToFetch, NULL);

// Инициализация очереди задач
srand(rank + time(NULL));
initializeJobs(rank);

// Создание потоков
pthread_t d_worker, d_fetcher, d_responder;
pthread_attr_t attr;
pthread_attr_init(&attr);

int args[] = {size, rank};

pthread_create(&d_worker, &attr, worker, args);
pthread_create(&d_fetcher, &attr, fetcher, args);
pthread_create(&d_responder, &attr, responder, args);

// Ожидание завершения потоков
pthread_join(d_worker, NULL);
pthread_join(d_fetcher, NULL);
pthread_join(d_responder, NULL);

// Освобождение ресурсов
pthread_mutex_destroy(&mutex);
pthread_cond_destroy(&c_jobAdded);
pthread_cond_destroy(&c_needToFetch);

MPI_Finalize();

std::cout << "Process " << rank << ": All done" << std::endl;
return 0;

  

}
