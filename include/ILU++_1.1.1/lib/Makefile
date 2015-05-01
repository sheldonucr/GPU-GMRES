# ################################################################################################################################## #
#                                                                                                                                    #
#                                                                                                                                    #
#   Makefile for iluplusplus/lib                                                                                                     #
#                                                                                                                                    #
#                                                                                                                                    #
# ################################################################################################################################## #


include ../Makefile.in


# ##################################################################################################################################

# for making the library libiluplusplus.a

LIB_OBJECTS     = iohb.o iluplusplus_interface.o

.cpp.o: Makefile
	$(CC) -c $(COMPILEOPTIONS) $(DEFS) $(INCLUDE) $<

../libiluplusplus$(VERSION).a:  Makefile $(LIB_OBJECTS)
	$(AR) $@ $(LIB_OBJECTS)

clean:
	rm -f $(OBJECTS) *.o; rm -f *~

realclean:
	rm -f *.o ; rm -f *~ ; rm -f ../libiluplusplus$(VERSION).a

# ##################################################################################################################################


iohb.o:                  iohb.h Makefile ../Makefile.in ../Makefile

iluplusplus_interface.o: iluplusplus.h iluplusplus_declarations.h declarations.h arrays.h functions.h function_class.h orderings.h \
                         iohb.h preconditioner.h iterative_solvers.h solving_routines.h iluplusplus_implementation.h               \
                         functions_implementation.h arrays_implementation.h function_class_implementation.h                        \
                         orderings_implementation.h parameters_implementation.h sparse_implementation.h                            \
                         preconditioner_implementation.h iterative_solvers_implementation.h solving_routines_implementation.h      \
                         iluplusplus_interface.h iluplusplus_interface.h parameters.h sparse.h solving_routines.h                  \
                         pmwm_declarations.h pmwm_implementation.h                                                                 \
                         Makefile ../Makefile.in ../Makefile

# ##################################################################################################################################