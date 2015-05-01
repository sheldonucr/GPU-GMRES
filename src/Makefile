#######################################################
#
#    Cadence Extended Truncated Balanced Realization
#                (*** CadETBR ***)
#
#######################################################
#
#     Authors: Duo Li, Zao Liu, Xue-Xin Liu, Kai He and Sheldon Tan
#
#     Functions: Makefile
#
#######################################################

CUDA_SM := -arch sm_35

MAIN = etbr_cmd 
MNAMAIN = mna_cmd

#Please install the follwing packages and modify the paths
ITPP = /home/locker/EE/mscad/num_lib/itpp-4.0.6
UMFPACK = /home/locker/EE/mscad/num_lib/umfpack/UMFPACK
CXSPARSE = /home/locker/EE/mscad/num_lib/CXSparse
UFCONFIG = /home/locker/EE/mscad/num_lib/umfpack/UFconfig
AMD = /home/locker/EE/mscad/num_lib/umfpack/AMD
PATOH = /home/locker/EE/mscad/num_lib/patoh
METIS = /home/locker/EE/mscad/num_lib/metis-5.0pre2
ITPPEX = /home/locker/EE/mscad/num_lib/it++external-3.0.0

cudaroot  := /usr/local/cuda
cudaSDKroot := /usr/local/cuda/samples
cusp_paths := -I../include/cusplibrary-0.5.1

cuINC = -I$(cudaroot)/include -I$(cudaSDKroot)/common/inc 
cuLIB = -L$(cudaroot)/lib64 -L$(cudaSDKroot)/lib -lcudart -L/usr/lib64 -lcuda  -lcublas -lcusparse 
INCFLAGS = -I$(ITPP) -I$(UMFPACK)/Include -I$(UMFPACK)/Source -I$(UFCONFIG) \
	-I$(AMD)/Include -I$(CXSPARSE)/Include -I$(PATOH) \
	-I$(METIS)/include  -I$(METIS)/libmetis  -I$(METIS)/GKlib/trunk \
	$(cuINC) -I../include/ILU++_1.1.1/lib


LIBFLAGS = -L$(ITPP)/lib -L$(UMFPACK)/Lib -L$(AMD)/Lib -L/usr/lib64 -lgfortran

LIBDIR = /home/locker/EE/mscad/num_lib
LIBS = $(LIBDIR)/libumfpack.a $(LIBDIR)/libamd.a $(LIBDIR)/libcxsparse.a \
	$(METIS)/build/Linux-x86_64/libmetis.a \
	$(METIS)/GKlib/builds/Linux-x86_64/libGKlib.a \
	-lm


SRCS = itpp_operations.cpp transim.cpp transim2.cpp etbr_dd.cpp form_dd.cpp solve_dd.cpp dd_save_load.cpp \
	partition.cpp partition3.cpp xgraph.cpp \
	ir_analysis.cpp dc_solver.cpp etbr.cpp etbr2.cpp itpp2csparse.cpp interp.cpp svd0.cpp \
	namepool.cpp hashtable.cpp element.cpp circuit.cpp matrix.cpp parser.cpp mna.cpp \
	etbr_thread.cpp etbr_wrapper.cpp mna_solve.cpp gpu_transim.cpp gpu_etbr_thread.cpp \
	mna_solve_gpu_gmres.cpp \
	SpMV_compute.cpp SpMV_inspect.cpp \
	iluk.cpp itsol.cpp formatConvert.cpp

#
CU_SRCS = cudaTranSim.cu wrapperGPUforPG.cu wrapperGMRESforPG.cu gmres_interface_pg.cu \
	kernels.cu gmres.cu preconditioner.cu \
	leftILU.cu \
	SpMV_alloc.cu SpMV_kernel.cu mySpMatrix.cu \
	setGPUdevice.cu

MAINSRCS = etbr_cmd_short.cpp 
MNAMAINSRCS = mna_cmd.cpp 
OBJS = $(addsuffix .o, $(basename $(SRCS)))
MAINOBJS = $(addsuffix .o, $(basename $(MAINSRCS)))
MNAMAINOBJS = $(addsuffix .o, $(basename $(MNAMAINSRCS)))

CU_OBJS = $(addsuffix .o, $(basename $(CU_SRCS)))

nvcc        := nvcc

CPP = g++
THREADFLAGS = -pthread
DEBUGFLAGS = -O3

$(MAIN): $(OBJS) $(MAINOBJS) $(TMAINOBJS) $(MNAMAINOBJS) $(LIBS) $(CU_OBJS)
	@echo "Link  etbr_cmd ...."
	@$(CPP) $(INCFLAGS) $(LIBFLAGS) $(FLAGS_OPT) $(DEBUGFLAGS) $(THREADFLAGS) -o etbr_cmd $(OBJS) $(MAINOBJS) $(CU_OBJS) $(cuLIB) $(LIBS) \
	-L$(ITPP)/itpp/.libs/ -litpp -L$(ITPPEX)/lib -lfftw3 -L../include -llapack_LINUX -lblas -lgfortranbegin -lgfortran -lm \
	-L../include/ILU++_1.1.1 -liluplusplus-1.1


.cpp.o:
	$(CPP) $(INCFLAGS) $(LIBFLAGS) $(DEBUGFLAGS) $(cusp_paths) -c $<


%.o: %.cu
	$(nvcc) $(CUDA_SM) $(cuINC) $(cusp_paths) $(DEBUGFLAGS) -c $^ -o $@


cudaTranSim.o: cudaTranSim.cu
	nvcc $(CUDA_SM) $(INCFLAGS) $(DEBUGFLAGS) -c $< -o $@ $(cuLIB)


depend:
	makedepend $(INCFLAGS) -- $(SRCS) # DO NOT DELETE

clean:
	@rm -rf *.o *~
# DO NOT DELETE

etbr_dd.o: $(ITPP)/itpp/base/timing.h
etbr_dd.o: $(ITPP)/itpp/base/smat.h
etbr_dd.o: $(ITPP)/itpp/config.h
etbr_dd.o: $(ITPP)/itpp/base/svec.h
etbr_dd.o: $(ITPP)/itpp/base/vec.h
etbr_dd.o: $(ITPP)/itpp/base/itassert.h
etbr_dd.o: $(ITPP)/itpp/base/math/misc.h
etbr_dd.o: $(ITPP)/itpp/base/copy_vector.h
etbr_dd.o: $(ITPP)/itpp/base/blas.h
etbr_dd.o: $(ITPP)/itpp/base/binary.h
etbr_dd.o: $(ITPP)/itpp/base/factory.h
etbr_dd.o: $(ITPP)/itpp/base/mat.h
etbr_dd.o: $(ITPP)/itpp/base/math/min_max.h
etbr_dd.o: $(ITPP)/itpp/base/specmat.h
etbr_dd.o: $(ITPP)/itpp/base/algebra/lapack.h
etbr_dd.o: $(ITPP)/itpp/base/algebra/ls_solve.h
etbr_dd.o: $(ITPP)/itpp/base/algebra/svd.h
etbr_dd.o: $(ITPP)/itpp/signal/transforms.h
etbr_dd.o: $(ITPP)/itpp/base/matfunc.h
etbr_dd.o: $(ITPP)/itpp/base/math/log_exp.h
etbr_dd.o: $(ITPP)/itpp/base/help_functions.h
etbr_dd.o: $(ITPP)/itpp/base/math/elem_math.h
etbr_dd.o: $(ITPP)/itpp/base/converters.h
etbr_dd.o: $(UMFPACK)/Include/umfpack.h
etbr_dd.o: $(UFCONFIG)/UFconfig.h
etbr_dd.o: /usr/include/limits.h /usr/include/features.h
etbr_dd.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
etbr_dd.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
etbr_dd.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
etbr_dd.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_symbolic.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_numeric.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_solve.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_free_symbolic.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_free_numeric.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_defaults.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_qsymbolic.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_wsolve.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_triplet_to_col.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_col_to_triplet.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_transpose.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_scale.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_get_lunz.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_get_numeric.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_get_symbolic.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_save_numeric.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_load_numeric.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_save_symbolic.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_load_symbolic.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_get_determinant.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_status.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_info.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_control.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_matrix.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_triplet.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_vector.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_symbolic.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_numeric.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_report_perm.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_timer.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_tictoc.h
etbr_dd.o: $(AMD)/Include/amd.h
etbr_dd.o: $(UMFPACK)/Include/umfpack_global.h
etbr_dd.o: etbr.h $(CXSPARSE)/Include/cs.h
etbr_dd.o: /usr/include/stdlib.h /usr/include/sys/types.h
etbr_dd.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
etbr_dd.o: /usr/include/time.h /usr/include/endian.h
etbr_dd.o: /usr/include/bits/endian.h /usr/include/sys/select.h
etbr_dd.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
etbr_dd.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
etbr_dd.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
etbr_dd.o: /usr/include/math.h /usr/include/bits/huge_val.h
etbr_dd.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
etbr_dd.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
etbr_dd.o: /usr/include/wchar.h /usr/include/bits/wchar.h
etbr_dd.o: /usr/include/gconv.h /usr/include/bits/stdio_lim.h
etbr_dd.o: /usr/include/bits/sys_errlist.h /usr/include/complex.h
etbr_dd.o: /usr/include/bits/cmathcalls.h etbr_dd.h
etbr_dd.o: $(METIS)/include/metis.h
etbr_dd.o: /usr/include/stdint.h /usr/include/inttypes.h interp.h svd0.h
form_dd.o: $(ITPP)/itpp/base/timing.h
form_dd.o: $(ITPP)/itpp/base/mat.h
form_dd.o: $(ITPP)/itpp/config.h
form_dd.o: $(ITPP)/itpp/base/itassert.h
form_dd.o: $(ITPP)/itpp/base/math/misc.h
form_dd.o: $(ITPP)/itpp/base/factory.h
form_dd.o: $(ITPP)/itpp/base/binary.h
form_dd.o: $(ITPP)/itpp/base/vec.h
form_dd.o: $(ITPP)/itpp/base/copy_vector.h
form_dd.o: $(ITPP)/itpp/base/blas.h
form_dd.o: $(ITPP)/itpp/base/algebra/lapack.h
form_dd.o: $(ITPP)/itpp/base/algebra/ls_solve.h
form_dd.o: $(CXSPARSE)/Include/cs.h
form_dd.o: /usr/include/stdlib.h /usr/include/features.h
form_dd.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
form_dd.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
form_dd.o: /usr/include/sys/types.h /usr/include/bits/types.h
form_dd.o: /usr/include/bits/typesizes.h /usr/include/time.h
form_dd.o: /usr/include/endian.h /usr/include/bits/endian.h
form_dd.o: /usr/include/sys/select.h /usr/include/bits/select.h
form_dd.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
form_dd.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
form_dd.o: /usr/include/alloca.h /usr/include/limits.h
form_dd.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
form_dd.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
form_dd.o: /usr/include/math.h /usr/include/bits/huge_val.h
form_dd.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
form_dd.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
form_dd.o: /usr/include/wchar.h /usr/include/bits/wchar.h
form_dd.o: /usr/include/gconv.h /usr/include/bits/stdio_lim.h
form_dd.o: /usr/include/bits/sys_errlist.h /usr/include/complex.h
form_dd.o: /usr/include/bits/cmathcalls.h
form_dd.o: $(UFCONFIG)/UFconfig.h
form_dd.o: $(UMFPACK)/Include/umfpack.h
form_dd.o: $(UMFPACK)/Include/umfpack_symbolic.h
form_dd.o: $(UMFPACK)/Include/umfpack_numeric.h
form_dd.o: $(UMFPACK)/Include/umfpack_solve.h
form_dd.o: $(UMFPACK)/Include/umfpack_free_symbolic.h
form_dd.o: $(UMFPACK)/Include/umfpack_free_numeric.h
form_dd.o: $(UMFPACK)/Include/umfpack_defaults.h
form_dd.o: $(UMFPACK)/Include/umfpack_qsymbolic.h
form_dd.o: $(UMFPACK)/Include/umfpack_wsolve.h
form_dd.o: $(UMFPACK)/Include/umfpack_triplet_to_col.h
form_dd.o: $(UMFPACK)/Include/umfpack_col_to_triplet.h
form_dd.o: $(UMFPACK)/Include/umfpack_transpose.h
form_dd.o: $(UMFPACK)/Include/umfpack_scale.h
form_dd.o: $(UMFPACK)/Include/umfpack_get_lunz.h
form_dd.o: $(UMFPACK)/Include/umfpack_get_numeric.h
form_dd.o: $(UMFPACK)/Include/umfpack_get_symbolic.h
form_dd.o: $(UMFPACK)/Include/umfpack_save_numeric.h
form_dd.o: $(UMFPACK)/Include/umfpack_load_numeric.h
form_dd.o: $(UMFPACK)/Include/umfpack_save_symbolic.h
form_dd.o: $(UMFPACK)/Include/umfpack_load_symbolic.h
form_dd.o: $(UMFPACK)/Include/umfpack_get_determinant.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_status.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_info.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_control.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_matrix.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_triplet.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_vector.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_symbolic.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_numeric.h
form_dd.o: $(UMFPACK)/Include/umfpack_report_perm.h
form_dd.o: $(UMFPACK)/Include/umfpack_timer.h
form_dd.o: $(UMFPACK)/Include/umfpack_tictoc.h
form_dd.o: $(AMD)/Include/amd.h
form_dd.o: $(UMFPACK)/Include/umfpack_global.h
form_dd.o: etbr_dd.h etbr.h
form_dd.o: $(ITPP)/itpp/base/smat.h
form_dd.o: $(ITPP)/itpp/base/svec.h
form_dd.o: $(ITPP)/itpp/base/math/min_max.h
form_dd.o: $(METIS)/include/metis.h
form_dd.o: /usr/include/stdint.h /usr/include/inttypes.h
solve_dd.o: /usr/include/stdio.h /usr/include/features.h
solve_dd.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
solve_dd.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
solve_dd.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
solve_dd.o: /usr/include/libio.h /usr/include/_G_config.h
solve_dd.o: /usr/include/wchar.h /usr/include/bits/wchar.h
solve_dd.o: /usr/include/gconv.h /usr/include/bits/stdio_lim.h
solve_dd.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
solve_dd.o: /usr/include/sys/types.h /usr/include/time.h
solve_dd.o: /usr/include/endian.h /usr/include/bits/endian.h
solve_dd.o: /usr/include/sys/select.h /usr/include/bits/select.h
solve_dd.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
solve_dd.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
solve_dd.o: /usr/include/alloca.h
solve_dd.o: $(ITPP)/itpp/base/timing.h
solve_dd.o: $(ITPP)/itpp/base/mat.h
solve_dd.o: $(ITPP)/itpp/config.h
solve_dd.o: $(ITPP)/itpp/base/itassert.h
solve_dd.o: $(ITPP)/itpp/base/math/misc.h
solve_dd.o: $(ITPP)/itpp/base/factory.h
solve_dd.o: $(ITPP)/itpp/base/binary.h
solve_dd.o: $(ITPP)/itpp/base/vec.h
solve_dd.o: $(ITPP)/itpp/base/copy_vector.h
solve_dd.o: $(ITPP)/itpp/base/blas.h
solve_dd.o: $(ITPP)/itpp/base/algebra/lapack.h
solve_dd.o: $(ITPP)/itpp/base/algebra/ls_solve.h
solve_dd.o: $(CXSPARSE)/Include/cs.h
solve_dd.o: /usr/include/limits.h /usr/include/bits/posix1_lim.h
solve_dd.o: /usr/include/bits/local_lim.h /usr/include/linux/limits.h
solve_dd.o: /usr/include/bits/posix2_lim.h /usr/include/math.h
solve_dd.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
solve_dd.o: /usr/include/bits/mathcalls.h /usr/include/complex.h
solve_dd.o: /usr/include/bits/cmathcalls.h
solve_dd.o: $(UFCONFIG)/UFconfig.h
solve_dd.o: $(UMFPACK)/Include/umfpack.h
solve_dd.o: $(UMFPACK)/Include/umfpack_symbolic.h
solve_dd.o: $(UMFPACK)/Include/umfpack_numeric.h
solve_dd.o: $(UMFPACK)/Include/umfpack_solve.h
solve_dd.o: $(UMFPACK)/Include/umfpack_free_symbolic.h
solve_dd.o: $(UMFPACK)/Include/umfpack_free_numeric.h
solve_dd.o: $(UMFPACK)/Include/umfpack_defaults.h
solve_dd.o: $(UMFPACK)/Include/umfpack_qsymbolic.h
solve_dd.o: $(UMFPACK)/Include/umfpack_wsolve.h
solve_dd.o: $(UMFPACK)/Include/umfpack_triplet_to_col.h
solve_dd.o: $(UMFPACK)/Include/umfpack_col_to_triplet.h
solve_dd.o: $(UMFPACK)/Include/umfpack_transpose.h
solve_dd.o: $(UMFPACK)/Include/umfpack_scale.h
solve_dd.o: $(UMFPACK)/Include/umfpack_get_lunz.h
solve_dd.o: $(UMFPACK)/Include/umfpack_get_numeric.h
solve_dd.o: $(UMFPACK)/Include/umfpack_get_symbolic.h
solve_dd.o: $(UMFPACK)/Include/umfpack_save_numeric.h
solve_dd.o: $(UMFPACK)/Include/umfpack_load_numeric.h
solve_dd.o: $(UMFPACK)/Include/umfpack_save_symbolic.h
solve_dd.o: $(UMFPACK)/Include/umfpack_load_symbolic.h
solve_dd.o: $(UMFPACK)/Include/umfpack_get_determinant.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_status.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_info.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_control.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_matrix.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_triplet.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_vector.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_symbolic.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_numeric.h
solve_dd.o: $(UMFPACK)/Include/umfpack_report_perm.h
solve_dd.o: $(UMFPACK)/Include/umfpack_timer.h
solve_dd.o: $(UMFPACK)/Include/umfpack_tictoc.h
solve_dd.o: $(AMD)/Include/amd.h
solve_dd.o: $(UMFPACK)/Include/umfpack_global.h
solve_dd.o: etbr_dd.h etbr.h
solve_dd.o: $(ITPP)/itpp/base/smat.h
solve_dd.o: $(ITPP)/itpp/base/svec.h
solve_dd.o: $(ITPP)/itpp/base/math/min_max.h
solve_dd.o: $(METIS)/include/metis.h
solve_dd.o: /usr/include/stdint.h /usr/include/inttypes.h
dd_save_load.o: $(CXSPARSE)/Include/cs.h
dd_save_load.o: /usr/include/stdlib.h /usr/include/features.h
dd_save_load.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
dd_save_load.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
dd_save_load.o: /usr/include/sys/types.h /usr/include/bits/types.h
dd_save_load.o: /usr/include/bits/typesizes.h /usr/include/time.h
dd_save_load.o: /usr/include/endian.h /usr/include/bits/endian.h
dd_save_load.o: /usr/include/sys/select.h /usr/include/bits/select.h
dd_save_load.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
dd_save_load.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
dd_save_load.o: /usr/include/alloca.h /usr/include/limits.h
dd_save_load.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
dd_save_load.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
dd_save_load.o: /usr/include/math.h /usr/include/bits/huge_val.h
dd_save_load.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
dd_save_load.o: /usr/include/stdio.h /usr/include/libio.h
dd_save_load.o: /usr/include/_G_config.h /usr/include/wchar.h
dd_save_load.o: /usr/include/bits/wchar.h /usr/include/gconv.h
dd_save_load.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
dd_save_load.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h
dd_save_load.o: $(UFCONFIG)/UFconfig.h
dd_save_load.o: etbr_dd.h etbr.h
dd_save_load.o: $(ITPP)/itpp/base/smat.h
dd_save_load.o: $(ITPP)/itpp/config.h
dd_save_load.o: $(ITPP)/itpp/base/svec.h
dd_save_load.o: $(ITPP)/itpp/base/vec.h
dd_save_load.o: $(ITPP)/itpp/base/itassert.h
dd_save_load.o: $(ITPP)/itpp/base/math/misc.h
dd_save_load.o: $(ITPP)/itpp/base/copy_vector.h
dd_save_load.o: $(ITPP)/itpp/base/blas.h
dd_save_load.o: $(ITPP)/itpp/base/binary.h
dd_save_load.o: $(ITPP)/itpp/base/factory.h
dd_save_load.o: $(ITPP)/itpp/base/mat.h
dd_save_load.o: $(ITPP)/itpp/base/math/min_max.h
dd_save_load.o: $(ITPP)/itpp/base/timing.h
dd_save_load.o: $(METIS)/include/metis.h
dd_save_load.o: /usr/include/stdint.h /usr/include/inttypes.h
partition.o: $(CXSPARSE)/Include/cs.h
partition.o: /usr/include/stdlib.h /usr/include/features.h
partition.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
partition.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
partition.o: /usr/include/sys/types.h /usr/include/bits/types.h
partition.o: /usr/include/bits/typesizes.h /usr/include/time.h
partition.o: /usr/include/endian.h /usr/include/bits/endian.h
partition.o: /usr/include/sys/select.h /usr/include/bits/select.h
partition.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
partition.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
partition.o: /usr/include/alloca.h /usr/include/limits.h
partition.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
partition.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
partition.o: /usr/include/math.h /usr/include/bits/huge_val.h
partition.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
partition.o: /usr/include/stdio.h /usr/include/libio.h
partition.o: /usr/include/_G_config.h /usr/include/wchar.h
partition.o: /usr/include/bits/wchar.h /usr/include/gconv.h
partition.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
partition.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h
partition.o: $(UFCONFIG)/UFconfig.h
partition3.o: $(CXSPARSE)/Include/cs.h
partition3.o: /usr/include/stdlib.h /usr/include/features.h
partition3.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
partition3.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
partition3.o: /usr/include/sys/types.h /usr/include/bits/types.h
partition3.o: /usr/include/bits/typesizes.h /usr/include/time.h
partition3.o: /usr/include/endian.h /usr/include/bits/endian.h
partition3.o: /usr/include/sys/select.h /usr/include/bits/select.h
partition3.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
partition3.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
partition3.o: /usr/include/alloca.h /usr/include/limits.h
partition3.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
partition3.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
partition3.o: /usr/include/math.h /usr/include/bits/huge_val.h
partition3.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
partition3.o: /usr/include/stdio.h /usr/include/libio.h
partition3.o: /usr/include/_G_config.h /usr/include/wchar.h
partition3.o: /usr/include/bits/wchar.h /usr/include/gconv.h
partition3.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
partition3.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h
partition3.o: $(UFCONFIG)/UFconfig.h
partition3.o: $(METIS)/include/metis.h
partition3.o: /usr/include/stdint.h /usr/include/inttypes.h
xgraph.o: $(ITPP)/itpp/base/vec.h
xgraph.o: $(ITPP)/itpp/config.h
xgraph.o: $(ITPP)/itpp/base/itassert.h
xgraph.o: $(ITPP)/itpp/base/math/misc.h
xgraph.o: $(ITPP)/itpp/base/copy_vector.h
xgraph.o: $(ITPP)/itpp/base/blas.h
xgraph.o: $(ITPP)/itpp/base/binary.h
xgraph.o: $(ITPP)/itpp/base/factory.h
xgraph.o: $(ITPP)/itpp/base/mat.h
ir_analysis.o: $(ITPP)/itpp/base/mat.h
ir_analysis.o: $(ITPP)/itpp/config.h
ir_analysis.o: $(ITPP)/itpp/base/itassert.h
ir_analysis.o: $(ITPP)/itpp/base/math/misc.h
ir_analysis.o: $(ITPP)/itpp/base/factory.h
ir_analysis.o: $(ITPP)/itpp/base/binary.h
ir_analysis.o: $(ITPP)/itpp/base/vec.h
ir_analysis.o: $(ITPP)/itpp/base/copy_vector.h
ir_analysis.o: $(ITPP)/itpp/base/blas.h
ir_analysis.o: $(ITPP)/itpp/base/math/min_max.h
ir_analysis.o: $(ITPP)/itpp/base/matfunc.h
ir_analysis.o: $(ITPP)/itpp/base/math/log_exp.h
ir_analysis.o: $(ITPP)/itpp/base/help_functions.h
ir_analysis.o: $(ITPP)/itpp/base/math/elem_math.h
ir_analysis.o: $(ITPP)/itpp/base/converters.h
ir_analysis.o: $(ITPP)/itpp/base/sort.h
dc_solver.o: $(ITPP)/itpp/base/timing.h
dc_solver.o: $(ITPP)/itpp/base/mat.h
dc_solver.o: $(ITPP)/itpp/config.h
dc_solver.o: $(ITPP)/itpp/base/itassert.h
dc_solver.o: $(ITPP)/itpp/base/math/misc.h
dc_solver.o: $(ITPP)/itpp/base/factory.h
dc_solver.o: $(ITPP)/itpp/base/binary.h
dc_solver.o: $(ITPP)/itpp/base/vec.h
dc_solver.o: $(ITPP)/itpp/base/copy_vector.h
dc_solver.o: $(ITPP)/itpp/base/blas.h
dc_solver.o: $(ITPP)/itpp/base/specmat.h
dc_solver.o: $(ITPP)/itpp/base/algebra/lapack.h
dc_solver.o: $(ITPP)/itpp/base/algebra/ls_solve.h
dc_solver.o: $(ITPP)/itpp/signal/transforms.h
dc_solver.o: $(ITPP)/itpp/base/matfunc.h
dc_solver.o: $(ITPP)/itpp/base/math/log_exp.h
dc_solver.o: $(ITPP)/itpp/base/help_functions.h
dc_solver.o: $(ITPP)/itpp/base/math/elem_math.h
dc_solver.o: $(ITPP)/itpp/base/converters.h
dc_solver.o: $(UMFPACK)/Include/umfpack.h
dc_solver.o: $(UFCONFIG)/UFconfig.h
dc_solver.o: /usr/include/limits.h /usr/include/features.h
dc_solver.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
dc_solver.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
dc_solver.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
dc_solver.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
dc_solver.o: $(UMFPACK)/Include/umfpack_symbolic.h
dc_solver.o: $(UMFPACK)/Include/umfpack_numeric.h
dc_solver.o: $(UMFPACK)/Include/umfpack_solve.h
dc_solver.o: $(UMFPACK)/Include/umfpack_free_symbolic.h
dc_solver.o: $(UMFPACK)/Include/umfpack_free_numeric.h
dc_solver.o: $(UMFPACK)/Include/umfpack_defaults.h
dc_solver.o: $(UMFPACK)/Include/umfpack_qsymbolic.h
dc_solver.o: $(UMFPACK)/Include/umfpack_wsolve.h
dc_solver.o: $(UMFPACK)/Include/umfpack_triplet_to_col.h
dc_solver.o: $(UMFPACK)/Include/umfpack_col_to_triplet.h
dc_solver.o: $(UMFPACK)/Include/umfpack_transpose.h
dc_solver.o: $(UMFPACK)/Include/umfpack_scale.h
dc_solver.o: $(UMFPACK)/Include/umfpack_get_lunz.h
dc_solver.o: $(UMFPACK)/Include/umfpack_get_numeric.h
dc_solver.o: $(UMFPACK)/Include/umfpack_get_symbolic.h
dc_solver.o: $(UMFPACK)/Include/umfpack_save_numeric.h
dc_solver.o: $(UMFPACK)/Include/umfpack_load_numeric.h
dc_solver.o: $(UMFPACK)/Include/umfpack_save_symbolic.h
dc_solver.o: $(UMFPACK)/Include/umfpack_load_symbolic.h
dc_solver.o: $(UMFPACK)/Include/umfpack_get_determinant.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_status.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_info.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_control.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_matrix.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_triplet.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_vector.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_symbolic.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_numeric.h
dc_solver.o: $(UMFPACK)/Include/umfpack_report_perm.h
dc_solver.o: $(UMFPACK)/Include/umfpack_timer.h
dc_solver.o: $(UMFPACK)/Include/umfpack_tictoc.h
dc_solver.o: $(AMD)/Include/amd.h
dc_solver.o: $(UMFPACK)/Include/umfpack_global.h
dc_solver.o: etbr.h $(ITPP)/itpp/base/smat.h
dc_solver.o: $(ITPP)/itpp/base/svec.h
dc_solver.o: $(ITPP)/itpp/base/math/min_max.h
dc_solver.o: $(CXSPARSE)/Include/cs.h
dc_solver.o: /usr/include/stdlib.h /usr/include/sys/types.h
dc_solver.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
dc_solver.o: /usr/include/time.h /usr/include/endian.h
dc_solver.o: /usr/include/bits/endian.h /usr/include/sys/select.h
dc_solver.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
dc_solver.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
dc_solver.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
dc_solver.o: /usr/include/math.h /usr/include/bits/huge_val.h
dc_solver.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
dc_solver.o: /usr/include/stdio.h /usr/include/libio.h
dc_solver.o: /usr/include/_G_config.h /usr/include/wchar.h
dc_solver.o: /usr/include/bits/wchar.h /usr/include/gconv.h
dc_solver.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
dc_solver.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h etbr_dd.h
dc_solver.o: $(METIS)/include/metis.h
dc_solver.o: /usr/include/stdint.h /usr/include/inttypes.h
etbr.o: $(ITPP)/itpp/base/timing.h
etbr.o: $(ITPP)/itpp/base/smat.h
etbr.o: $(ITPP)/itpp/config.h
etbr.o: $(ITPP)/itpp/base/svec.h
etbr.o: $(ITPP)/itpp/base/vec.h
etbr.o: $(ITPP)/itpp/base/itassert.h
etbr.o: $(ITPP)/itpp/base/math/misc.h
etbr.o: $(ITPP)/itpp/base/copy_vector.h
etbr.o: $(ITPP)/itpp/base/blas.h
etbr.o: $(ITPP)/itpp/base/binary.h
etbr.o: $(ITPP)/itpp/base/factory.h
etbr.o: $(ITPP)/itpp/base/mat.h
etbr.o: $(ITPP)/itpp/base/math/min_max.h
etbr.o: $(ITPP)/itpp/base/specmat.h
etbr.o: $(ITPP)/itpp/base/algebra/lapack.h
etbr.o: $(ITPP)/itpp/base/algebra/ls_solve.h
etbr.o: $(ITPP)/itpp/base/algebra/svd.h
etbr.o: $(ITPP)/itpp/signal/transforms.h
etbr.o: $(ITPP)/itpp/base/matfunc.h
etbr.o: $(ITPP)/itpp/base/math/log_exp.h
etbr.o: $(ITPP)/itpp/base/help_functions.h
etbr.o: $(ITPP)/itpp/base/math/elem_math.h
etbr.o: $(ITPP)/itpp/base/converters.h
etbr.o: $(UMFPACK)/Include/umfpack.h
etbr.o: $(UFCONFIG)/UFconfig.h
etbr.o: /usr/include/limits.h /usr/include/features.h
etbr.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
etbr.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
etbr.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
etbr.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
etbr.o: $(UMFPACK)/Include/umfpack_symbolic.h
etbr.o: $(UMFPACK)/Include/umfpack_numeric.h
etbr.o: $(UMFPACK)/Include/umfpack_solve.h
etbr.o: $(UMFPACK)/Include/umfpack_free_symbolic.h
etbr.o: $(UMFPACK)/Include/umfpack_free_numeric.h
etbr.o: $(UMFPACK)/Include/umfpack_defaults.h
etbr.o: $(UMFPACK)/Include/umfpack_qsymbolic.h
etbr.o: $(UMFPACK)/Include/umfpack_wsolve.h
etbr.o: $(UMFPACK)/Include/umfpack_triplet_to_col.h
etbr.o: $(UMFPACK)/Include/umfpack_col_to_triplet.h
etbr.o: $(UMFPACK)/Include/umfpack_transpose.h
etbr.o: $(UMFPACK)/Include/umfpack_scale.h
etbr.o: $(UMFPACK)/Include/umfpack_get_lunz.h
etbr.o: $(UMFPACK)/Include/umfpack_get_numeric.h
etbr.o: $(UMFPACK)/Include/umfpack_get_symbolic.h
etbr.o: $(UMFPACK)/Include/umfpack_save_numeric.h
etbr.o: $(UMFPACK)/Include/umfpack_load_numeric.h
etbr.o: $(UMFPACK)/Include/umfpack_save_symbolic.h
etbr.o: $(UMFPACK)/Include/umfpack_load_symbolic.h
etbr.o: $(UMFPACK)/Include/umfpack_get_determinant.h
etbr.o: $(UMFPACK)/Include/umfpack_report_status.h
etbr.o: $(UMFPACK)/Include/umfpack_report_info.h
etbr.o: $(UMFPACK)/Include/umfpack_report_control.h
etbr.o: $(UMFPACK)/Include/umfpack_report_matrix.h
etbr.o: $(UMFPACK)/Include/umfpack_report_triplet.h
etbr.o: $(UMFPACK)/Include/umfpack_report_vector.h
etbr.o: $(UMFPACK)/Include/umfpack_report_symbolic.h
etbr.o: $(UMFPACK)/Include/umfpack_report_numeric.h
etbr.o: $(UMFPACK)/Include/umfpack_report_perm.h
etbr.o: $(UMFPACK)/Include/umfpack_timer.h
etbr.o: $(UMFPACK)/Include/umfpack_tictoc.h
etbr.o: $(AMD)/Include/amd.h
etbr.o: $(UMFPACK)/Include/umfpack_global.h
etbr.o: etbr.h $(CXSPARSE)/Include/cs.h
etbr.o: /usr/include/stdlib.h /usr/include/sys/types.h
etbr.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
etbr.o: /usr/include/time.h /usr/include/endian.h /usr/include/bits/endian.h
etbr.o: /usr/include/sys/select.h /usr/include/bits/select.h
etbr.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
etbr.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
etbr.o: /usr/include/alloca.h /usr/include/math.h
etbr.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
etbr.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
etbr.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
etbr.o: /usr/include/bits/wchar.h /usr/include/gconv.h
etbr.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
etbr.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h interp.h svd0.h
etbr2.o: $(ITPP)/itpp/base/timing.h
etbr2.o: $(ITPP)/itpp/base/smat.h
etbr2.o: $(ITPP)/itpp/config.h
etbr2.o: $(ITPP)/itpp/base/svec.h
etbr2.o: $(ITPP)/itpp/base/vec.h
etbr2.o: $(ITPP)/itpp/base/itassert.h
etbr2.o: $(ITPP)/itpp/base/math/misc.h
etbr2.o: $(ITPP)/itpp/base/copy_vector.h
etbr2.o: $(ITPP)/itpp/base/blas.h
etbr2.o: $(ITPP)/itpp/base/binary.h
etbr2.o: $(ITPP)/itpp/base/factory.h
etbr2.o: $(ITPP)/itpp/base/mat.h
etbr2.o: $(ITPP)/itpp/base/math/min_max.h
etbr2.o: $(ITPP)/itpp/base/specmat.h
etbr2.o: $(ITPP)/itpp/base/algebra/lapack.h
etbr2.o: $(ITPP)/itpp/base/algebra/lu.h
etbr2.o: $(ITPP)/itpp/base/algebra/ls_solve.h
etbr2.o: $(ITPP)/itpp/base/algebra/svd.h
etbr2.o: $(ITPP)/itpp/signal/transforms.h
etbr2.o: $(ITPP)/itpp/base/matfunc.h
etbr2.o: $(ITPP)/itpp/base/math/log_exp.h
etbr2.o: $(ITPP)/itpp/base/help_functions.h
etbr2.o: $(ITPP)/itpp/base/math/elem_math.h
etbr2.o: $(ITPP)/itpp/base/converters.h
etbr2.o: $(UMFPACK)/Include/umfpack.h
etbr2.o: $(UFCONFIG)/UFconfig.h
etbr2.o: /usr/include/limits.h /usr/include/features.h
etbr2.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
etbr2.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
etbr2.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
etbr2.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
etbr2.o: $(UMFPACK)/Include/umfpack_symbolic.h
etbr2.o: $(UMFPACK)/Include/umfpack_numeric.h
etbr2.o: $(UMFPACK)/Include/umfpack_solve.h
etbr2.o: $(UMFPACK)/Include/umfpack_free_symbolic.h
etbr2.o: $(UMFPACK)/Include/umfpack_free_numeric.h
etbr2.o: $(UMFPACK)/Include/umfpack_defaults.h
etbr2.o: $(UMFPACK)/Include/umfpack_qsymbolic.h
etbr2.o: $(UMFPACK)/Include/umfpack_wsolve.h
etbr2.o: $(UMFPACK)/Include/umfpack_triplet_to_col.h
etbr2.o: $(UMFPACK)/Include/umfpack_col_to_triplet.h
etbr2.o: $(UMFPACK)/Include/umfpack_transpose.h
etbr2.o: $(UMFPACK)/Include/umfpack_scale.h
etbr2.o: $(UMFPACK)/Include/umfpack_get_lunz.h
etbr2.o: $(UMFPACK)/Include/umfpack_get_numeric.h
etbr2.o: $(UMFPACK)/Include/umfpack_get_symbolic.h
etbr2.o: $(UMFPACK)/Include/umfpack_save_numeric.h
etbr2.o: $(UMFPACK)/Include/umfpack_load_numeric.h
etbr2.o: $(UMFPACK)/Include/umfpack_save_symbolic.h
etbr2.o: $(UMFPACK)/Include/umfpack_load_symbolic.h
etbr2.o: $(UMFPACK)/Include/umfpack_get_determinant.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_status.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_info.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_control.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_matrix.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_triplet.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_vector.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_symbolic.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_numeric.h
etbr2.o: $(UMFPACK)/Include/umfpack_report_perm.h
etbr2.o: $(UMFPACK)/Include/umfpack_timer.h
etbr2.o: $(UMFPACK)/Include/umfpack_tictoc.h
etbr2.o: $(AMD)/Include/amd.h
etbr2.o: $(UMFPACK)/Include/umfpack_global.h
etbr2.o: etbr.h $(CXSPARSE)/Include/cs.h
etbr2.o: /usr/include/stdlib.h /usr/include/sys/types.h
etbr2.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
etbr2.o: /usr/include/time.h /usr/include/endian.h /usr/include/bits/endian.h
etbr2.o: /usr/include/sys/select.h /usr/include/bits/select.h
etbr2.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
etbr2.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
etbr2.o: /usr/include/alloca.h /usr/include/math.h
etbr2.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
etbr2.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
etbr2.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
etbr2.o: /usr/include/bits/wchar.h /usr/include/gconv.h
etbr2.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
etbr2.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h interp.h
etbr2.o: svd0.h
itpp2csparse.o: $(ITPP)/itpp/base/smat.h
itpp2csparse.o: $(ITPP)/itpp/config.h
itpp2csparse.o: $(ITPP)/itpp/base/svec.h
itpp2csparse.o: $(ITPP)/itpp/base/vec.h
itpp2csparse.o: $(ITPP)/itpp/base/itassert.h
itpp2csparse.o: $(ITPP)/itpp/base/math/misc.h
itpp2csparse.o: $(ITPP)/itpp/base/copy_vector.h
itpp2csparse.o: $(ITPP)/itpp/base/blas.h
itpp2csparse.o: $(ITPP)/itpp/base/binary.h
itpp2csparse.o: $(ITPP)/itpp/base/factory.h
itpp2csparse.o: $(ITPP)/itpp/base/mat.h
itpp2csparse.o: $(ITPP)/itpp/base/math/min_max.h
itpp2csparse.o: $(CXSPARSE)/Include/cs.h
itpp2csparse.o: /usr/include/stdlib.h /usr/include/features.h
itpp2csparse.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
itpp2csparse.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
itpp2csparse.o: /usr/include/sys/types.h /usr/include/bits/types.h
itpp2csparse.o: /usr/include/bits/typesizes.h /usr/include/time.h
itpp2csparse.o: /usr/include/endian.h /usr/include/bits/endian.h
itpp2csparse.o: /usr/include/sys/select.h /usr/include/bits/select.h
itpp2csparse.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
itpp2csparse.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
itpp2csparse.o: /usr/include/alloca.h /usr/include/limits.h
itpp2csparse.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
itpp2csparse.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
itpp2csparse.o: /usr/include/math.h /usr/include/bits/huge_val.h
itpp2csparse.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
itpp2csparse.o: /usr/include/stdio.h /usr/include/libio.h
itpp2csparse.o: /usr/include/_G_config.h /usr/include/wchar.h
itpp2csparse.o: /usr/include/bits/wchar.h /usr/include/gconv.h
itpp2csparse.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
itpp2csparse.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h
itpp2csparse.o: $(UFCONFIG)/UFconfig.h
interp.o: $(ITPP)/itpp/base/mat.h
interp.o: $(ITPP)/itpp/config.h
interp.o: $(ITPP)/itpp/base/itassert.h
interp.o: $(ITPP)/itpp/base/math/misc.h
interp.o: $(ITPP)/itpp/base/factory.h
interp.o: $(ITPP)/itpp/base/binary.h
interp.o: $(ITPP)/itpp/base/vec.h
interp.o: $(ITPP)/itpp/base/copy_vector.h
interp.o: $(ITPP)/itpp/base/blas.h
interp.o: $(ITPP)/itpp/base/math/elem_math.h
interp.o: $(ITPP)/itpp/base/help_functions.h
interp.o: $(ITPP)/itpp/base/converters.h
interp.o: /usr/include/assert.h /usr/include/features.h
interp.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
interp.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h interp.h
svd0.o: svd0.h $(ITPP)/itpp/base/vec.h
svd0.o: $(ITPP)/itpp/config.h
svd0.o: $(ITPP)/itpp/base/itassert.h
svd0.o: $(ITPP)/itpp/base/math/misc.h
svd0.o: $(ITPP)/itpp/base/copy_vector.h
svd0.o: $(ITPP)/itpp/base/blas.h
svd0.o: $(ITPP)/itpp/base/binary.h
svd0.o: $(ITPP)/itpp/base/factory.h
svd0.o: $(ITPP)/itpp/base/mat.h
svd0.o: $(ITPP)/itpp/base/algebra/lapack.h
namepool.o: namepool.h
hashtable.o: hashtable.h namepool.h /usr/include/stdlib.h
hashtable.o: /usr/include/features.h /usr/include/sys/cdefs.h
hashtable.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
hashtable.o: /usr/include/gnu/stubs-64.h /usr/include/sys/types.h
hashtable.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
hashtable.o: /usr/include/time.h /usr/include/endian.h
hashtable.o: /usr/include/bits/endian.h /usr/include/sys/select.h
hashtable.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
hashtable.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
hashtable.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
element.o: element.h namepool.h /usr/include/stdio.h /usr/include/features.h
element.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
element.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
element.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
element.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
element.o: /usr/include/bits/wchar.h /usr/include/gconv.h
element.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
element.o: hashtable.h
circuit.o: circuit.h namepool.h element.h /usr/include/stdio.h
circuit.o: /usr/include/features.h /usr/include/sys/cdefs.h
circuit.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
circuit.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
circuit.o: /usr/include/bits/typesizes.h /usr/include/libio.h
circuit.o: /usr/include/_G_config.h /usr/include/wchar.h
circuit.o: /usr/include/bits/wchar.h /usr/include/gconv.h
circuit.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
circuit.o: hashtable.h /usr/include/stdlib.h /usr/include/sys/types.h
circuit.o: /usr/include/time.h /usr/include/endian.h
circuit.o: /usr/include/bits/endian.h /usr/include/sys/select.h
circuit.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
circuit.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
circuit.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
matrix.o: matrix.h $(ITPP)/itpp/base/smat.h
matrix.o: $(ITPP)/itpp/config.h
matrix.o: $(ITPP)/itpp/base/svec.h
matrix.o: $(ITPP)/itpp/base/vec.h
matrix.o: $(ITPP)/itpp/base/itassert.h
matrix.o: $(ITPP)/itpp/base/math/misc.h
matrix.o: $(ITPP)/itpp/base/copy_vector.h
matrix.o: $(ITPP)/itpp/base/blas.h
matrix.o: $(ITPP)/itpp/base/binary.h
matrix.o: $(ITPP)/itpp/base/factory.h
matrix.o: $(ITPP)/itpp/base/mat.h
matrix.o: $(ITPP)/itpp/base/math/min_max.h
matrix.o: $(CXSPARSE)/Include/cs.h
matrix.o: /usr/include/stdlib.h /usr/include/features.h
matrix.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
matrix.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
matrix.o: /usr/include/sys/types.h /usr/include/bits/types.h
matrix.o: /usr/include/bits/typesizes.h /usr/include/time.h
matrix.o: /usr/include/endian.h /usr/include/bits/endian.h
matrix.o: /usr/include/sys/select.h /usr/include/bits/select.h
matrix.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
matrix.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
matrix.o: /usr/include/alloca.h /usr/include/limits.h
matrix.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
matrix.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
matrix.o: /usr/include/math.h /usr/include/bits/huge_val.h
matrix.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
matrix.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
matrix.o: /usr/include/wchar.h /usr/include/bits/wchar.h /usr/include/gconv.h
matrix.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
matrix.o: /usr/include/complex.h /usr/include/bits/cmathcalls.h
matrix.o: $(UFCONFIG)/UFconfig.h
parser.o: /usr/include/stdio.h /usr/include/features.h
parser.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
parser.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
parser.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
parser.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
parser.o: /usr/include/bits/wchar.h /usr/include/gconv.h
parser.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
parser.o: /usr/include/stdlib.h /usr/include/sys/types.h /usr/include/time.h
parser.o: /usr/include/endian.h /usr/include/bits/endian.h
parser.o: /usr/include/sys/select.h /usr/include/bits/select.h
parser.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
parser.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
parser.o: /usr/include/alloca.h /usr/include/math.h
parser.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
parser.o: /usr/include/bits/mathcalls.h parser.h circuit.h namepool.h
parser.o: element.h hashtable.h etbr.h
parser.o: $(ITPP)/itpp/base/smat.h
parser.o: $(ITPP)/itpp/config.h
parser.o: $(ITPP)/itpp/base/svec.h
parser.o: $(ITPP)/itpp/base/vec.h
parser.o: $(ITPP)/itpp/base/itassert.h
parser.o: $(ITPP)/itpp/base/math/misc.h
parser.o: $(ITPP)/itpp/base/copy_vector.h
parser.o: $(ITPP)/itpp/base/blas.h
parser.o: $(ITPP)/itpp/base/binary.h
parser.o: $(ITPP)/itpp/base/factory.h
parser.o: $(ITPP)/itpp/base/mat.h
parser.o: $(ITPP)/itpp/base/math/min_max.h
parser.o: $(CXSPARSE)/Include/cs.h
parser.o: /usr/include/limits.h /usr/include/bits/posix1_lim.h
parser.o: /usr/include/bits/local_lim.h /usr/include/linux/limits.h
parser.o: /usr/include/bits/posix2_lim.h /usr/include/complex.h
parser.o: /usr/include/bits/cmathcalls.h
parser.o: $(UFCONFIG)/UFconfig.h matrix.h
mna.o: mna.h circuit.h namepool.h element.h /usr/include/stdio.h
mna.o: /usr/include/features.h /usr/include/sys/cdefs.h
mna.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
mna.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
mna.o: /usr/include/bits/typesizes.h /usr/include/libio.h
mna.o: /usr/include/_G_config.h /usr/include/wchar.h
mna.o: /usr/include/bits/wchar.h /usr/include/gconv.h
mna.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
mna.o: hashtable.h matrix.h
mna.o: $(ITPP)/itpp/base/smat.h
mna.o: $(ITPP)/itpp/config.h
mna.o: $(ITPP)/itpp/base/svec.h
mna.o: $(ITPP)/itpp/base/vec.h
mna.o: $(ITPP)/itpp/base/itassert.h
mna.o: $(ITPP)/itpp/base/math/misc.h
mna.o: $(ITPP)/itpp/base/copy_vector.h
mna.o: $(ITPP)/itpp/base/blas.h
mna.o: $(ITPP)/itpp/base/binary.h
mna.o: $(ITPP)/itpp/base/factory.h
mna.o: $(ITPP)/itpp/base/mat.h
mna.o: $(ITPP)/itpp/base/math/min_max.h
mna.o: $(CXSPARSE)/Include/cs.h /usr/include/stdlib.h
mna.o: /usr/include/sys/types.h /usr/include/time.h /usr/include/endian.h
mna.o: /usr/include/bits/endian.h /usr/include/sys/select.h
mna.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
mna.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
mna.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
mna.o: /usr/include/limits.h /usr/include/bits/posix1_lim.h
mna.o: /usr/include/bits/local_lim.h /usr/include/linux/limits.h
mna.o: /usr/include/bits/posix2_lim.h /usr/include/math.h
mna.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
mna.o: /usr/include/bits/mathcalls.h /usr/include/complex.h
mna.o: /usr/include/bits/cmathcalls.h
mna.o: $(UFCONFIG)/UFconfig.h
