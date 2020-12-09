HIP_PATH    ?=/opt/rocm-3.9.0
HIPCC       :=$(HIP_PATH)/bin/hipcc
INCLUDE     :=-I/home/summit/.local/cfitsio/include\
	      -I/home/summit/.local/wcslib/include/wcslib\
	      -I./
LIBRARIES   :=-L/home/summit/.local/cfitsio/lib\
              -L/home/summit/.local/wcslib/lib

HIPCC_FLAGS := -fgpu-rdc --hip-link -Wno-deprecated-register
#HIPCC_LINK_FLAGS := --hip-link -fgpu-rdc
CXX_FLAGS := -std=c++11
FITS_FLAGS := -lm -lcfitsio -lwcs

helpers.o: helpers.cu
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

healpix.o: healpix.cu
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

gmap.o: gmap.cpp
	$(HIPCC) -O3 $(HIPCC_FLAGS) $(INCLUDE) -o $@ -c $<

gridding.o: oleder.cu
	$(HIPCC) -O3 $(HIPCC_FLAGS) $(INCLUDE) -o $@ -c $<

main.o: main.cpp
	$(HIPCC) -O3 $(HIPCC_FLAGS)  -o $@ -c $<

HCGrid: helpers.o healpix.o gmap.o gridding.o main.o
	@ echo ./$@ $+
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ $+ $(LIBRARIES) $(FITS_FLAGS)

clean:
	rm -rf *.o HCGrid
