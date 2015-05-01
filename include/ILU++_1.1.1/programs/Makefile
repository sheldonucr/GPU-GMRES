# ################################################################################################################################## #
#                                                                                                                                    #
#                                                                                                                                    #
#   Makefile for iluplusplus/programs                                                                                                #
#                                                                                                                                    #
#                                                                                                                                    #
# ################################################################################################################################## #


include ../Makefile.in

INCLUDE = -I ../lib/

ILUPLUSPLUSLIBPATH =  -L ../

OBJECTS = main.o main_solve.o main_templates.o

default:   ../main ../main_solve ../main_templates

.cpp.o: Makefile ../Makefile ../Makefile.in
	$(CC) -c $(INCLUDE) $(COMPILEOPTIONS) $(DEFS) $<

../main: main.o ../libiluplusplus$(VERSION).a Makefile ../Makefile ../Makefile.in
	$(CC) -o $@ main.o $(LINKOPTIONS)  $(ILUPLUSPLUSLIBPATH)

../main_solve: main_solve.o ../libiluplusplus$(VERSION).a Makefile ../Makefile ../Makefile.in
	$(CC) -o $@ main_solve.o $(LINKOPTIONS)  $(ILUPLUSPLUSLIBPATH)

../main_templates: main_templates.o ../lib/iohb.o Makefile ../Makefile ../Makefile.in
	$(CC) -o $@ main_templates.o ../lib/iohb.o $(LINKOPTIONS_TEM)

clean:
	rm -f $(OBJECTS) *.o ; rm -f *~ ;

realclean:
	rm -f $(OBJECTS) *.o ; rm -f *~ ;


# ##################################################################################################################################

main.o:                       ../lib/iluplusplus_interface.h Makefile ../Makefile ../Makefile.in

main_solve.o:                 ../lib/iluplusplus_interface.h Makefile ../Makefile ../Makefile.in


main_templates.o:             ../lib/iluplusplus.h ../lib/iluplusplus_declarations.h ../lib/declarations.h   \
                              ../lib/arrays.h ../lib/functions.h  ../lib/function_class.h ../lib/orderings.h \
                              ../lib/iohb.h ../lib/preconditioner.h ../lib/iterative_solvers.h               \
                              ../lib/solving_routines.h ../lib/iluplusplus_implementation.h                  \
                              ../lib/functions_implementation.h ../lib/arrays_implementation.h               \
                              ../lib/function_class_implementation.h  ../lib/orderings_implementation.h      \
                              ../lib/parameters_implementation.h ../lib/sparse_implementation.h              \
                              ../lib/preconditioner_implementation.h                                         \
                              ../lib/iterative_solvers_implementation.h                                      \
                              ../lib/solving_routines_implementation.h  ../lib/iluplusplus_interface.h       \
                              ../lib/iluplusplus_interface.h ../lib/parameters.h ../lib/sparse.h             \
                              ../lib/pmwm_declarations.h ../lib/pmwm_implementation.h                        \
                              ../lib/solving_routines.h Makefile ../Makefile ../Makefile.in


# ##################################################################################################################################