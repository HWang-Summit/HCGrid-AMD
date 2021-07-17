HIP_PATH    ?=/opt/rocm
HIPCC       :=$(HIP_PATH)/bin/hipcc
INCLUDE     :=-I/home/summit/.local/cfitsio/include\
	      -I/home/summit/.local/wcslib/include/wcslib\
	      -I./
LIBRARIES   :=-L/home/summit/.local/cfitsio/lib\
              -L/home/summit/.local/wcslib/lib

HIPCC_FLAGS := -fgpu-rdc -Wno-deprecated-register
HIPCC_LINK_FLAGS := -fgpu-rdc --hip-link
CXX_FLAGS := -std=c++11
FITS_FLAGS := -lm -lcfitsio -lwcs

HCGrid: helpers.o healpix.o gmap.o gridding.o main.o
	$(HIPCC) -O3 $(HIPCC_LINK_FLAGS) -o $@ $+ $(LIBRARIES) $(FITS_FLAGS)

helpers.o: helpers.hip
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

healpix.o: healpix.hip
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

gmap.o: gmap.cpp
	$(HIPCC) -O3 $(HIPCC_FLAGS) $(INCLUDE) -o $@ -c $<

gridding.o: gridding.hip
	$(HIPCC) -O3 $(HIPCC_FLAGS) $(INCLUDE) -o $@ -c $<

main.o: HCGrid.cpp
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

clean:
	rm -rf *.o HCGrid
