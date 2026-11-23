// main.cpp — md3_engine 入口：test / run
#include <cstring>
#include <cstdio>

int run_ops_tests();
int run_engine(int argc, char **argv);

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "test") == 0) {
        return run_ops_tests();
    }
    return run_engine(argc, argv);
}
