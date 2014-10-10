CFLAGS += 

a.out: *.cu *.cpp
	nvcc -I$(CUDA_PATH)/samples/common/inc/ $^ -arch=sm_20

clean:
	rm -f *.txt *.out *.png

plot: 
	$(SHELL) ./unshared.sh
	$(SHELL) ./shared.sh
	$(SHELL) ./all.sh

.PHONY: clean
