# vim:comments=
#
# I don't like the Cuda SDK makefiles. Here's my own! This makefile should
# allow you to build Cuda programs independent of the NVidia Cuda SDK. Its
# simplicity makes it easy for you to put objects, binaries, etc where you want
# them. There is even some boiler-plate Makefile magic that you can copy
# into other Makefiles.
#
# Mark Murphy 2008

# Fill in the name of the output binary here
target    := gmres_SpMV

# List of sources, with .c, .cu, and .cc extensions
sources   := ainv.cu preconditioner.cu main1.cu SpMV_alloc.cu SpMV_compute.cpp SpMV_kernel.cu gmres.cu \
	SpMV_gen.cpp SpMV_inspect.cpp leftILU.cpp mySpMatrix.cu

# Other things that need to be built, e.g. .cubin files
extradeps := 


# Flags common to all compilers. You can set these on the comamnd line, e.g:
# $ make opt="" dbg="-g" warn="-Wno-deptrcated-declarations -Wall -Werror"

opt  ?= -O3
dbg  ?= 
warn ?= -Wall -Werror


# This is where the cuda runtime libraries and includes reside on the
# berkeley millennium/rad clusters. It may be different on your machine.

cudaroot  := /usr/local/cuda
#cudaSDKroot := /home/eeres/kzhai/NVIDIA_GPU_Computing_SDK/C
cudaSDKroot := /usr/local/cuda/samples

#----- C compilation options ------
gcc        := /usr/bin/gcc
cflags     += $(opt) $(dbg) $(warn)
clib_paths :=
cinc_paths := 
clibraries := 


#----- C++ compilation options ------
gpp         := /usr/bin/g++
ccflags     += $(opt) $(dbg) $(warn)
cclib_paths :=
ccinc_paths := 
#-I$(cudaroot)/include  -I$(cudaSDKroot)/common/inc
cclibraries := 

#----- CUDA compilation options -----

nvcc        := nvcc
cuflags     += $(opt) $(dbg) -arch sm_20
culib_paths := -L$(cudaroot)/lib64 -L$(cudaSDKroot)/lib
# libcutil.a  libparamgl.a  librendercheckgl.a
cuinc_paths := -I$(cudaroot)/include -I$(cudaSDKroot)/common/inc 
culibraries := -lcuda -lcudart -lcublas -lcusparse

#cusp_paths := -I../include/cusplibrary-0.5.1
cusp_paths := -I../include/cusp-v0.3.1

#culibraries := -lcuda -lcudart -lcutil_x86_64 -lcublas -lcusparse

#
lib_paths   := $(culib_paths) $(cclib_paths) $(clib_paths)
libraries   := $(culibraries) $(cclibraries) $(clibraries)


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and Cuda files.

csources  := $(filter %.c,  $(sources))
ccsources := $(filter %.cc, $(sources)) \
	     $(filter %.cpp,$(sources))
cusources := $(filter %.cu, $(sources))

# This code generates a list of object files by replacing filename extensions

objects := $(patsubst %.c,  %.o,$(csources))  \
           $(patsubst %.cu, %.o,$(cusources)) \
	   $(patsubst %.cpp,%.o,$(filter %.cpp,$(ccsources))) \
	   $(patsubst %.cc, %.o,$(filter %.cc, $(ccsources)))


#----- Build rules ------

$(target): $(extradeps) 


$(target): $(objects) 
	$(gpp) $(objects) $(lib_paths) $(libraries) -o $@ 

%.o: %.cu
	$(nvcc) -c $^ $(cuflags) $(cuinc_paths) $(cusp_paths) -o $@ 
#	echo $(objects)

%.cubin: %.cu
	$(nvcc) -cubin $(cuflags) $(cuinc_paths) $(cusp_paths) $^

%.o: %.cc %.cpp
	$(gpp) -c $^ $(ccflags) $(ccinc_paths) -o $@

%.o: %.c
	$(gcc) -c $^ $(cflags) $(cinc_paths) -o $@

clean:
	rm -f *.o $(target) makefile.*dep


#----- Dependency Generation -----
#
# If a particular set of sources is non-empty, then have rules for
# generating the necessary dep files.
#

ccdep := ccdep.mk
cdep  := cdep.mk
cudep := cudep.mk


depfiles =

ifneq ($(ccsources),)

depfiles += $(ccdep)
$(ccdep): $(ccsources)
	$(gpp) -MM $(ccsources) > $(ccdep)

else

$(ccdep):

endif

ifneq ($(cusources),)

depfiles += $(cudep)
$(cudep):
	$(gpp) -MM -x c++ $(cusources) > $(cudep)

else

$(cudep):

endif

ifneq ($(csources),)

depfiles += $(cdep)
$(cdep): $(csources)
	$(gcc) -MM -x c $(csources) > $(cdep)

else

$(cdep):

endif

.PHONY: dep
dep: $(depfiles)


