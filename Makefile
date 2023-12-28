all: clang.s # gcc.s

clang.s: sketch.cc
	clang-18 --target=riscv64-linux-gnu -std=c++17 -pedantic -Wall -Werror -O3 -march=rv64gcv_zba_zbb_zbs -S $< -o $@

gcc.s: sketch.cc
	riscv64-linux-gnu-gcc-12 -std=c++17 -pedantic -Wall -Werror -O3 -march=rv64gcv_zvl512b_zba_zbb_zbs -Wno-psabi -S $< -o $@
