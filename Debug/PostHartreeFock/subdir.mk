################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../PostHartreeFock/Create1ParticleOperator.cc \
../PostHartreeFock/Create2ParticleOperator.cc 

CPP_SRCS += \
../PostHartreeFock/BasisTransform.cpp \
../PostHartreeFock/BuildTTOperator.cpp \
../PostHartreeFock/BuildTTOperator_molecular.cpp \
../PostHartreeFock/BuildTTOperator_parallel.cpp \
../PostHartreeFock/TestBuildTTOperator_with_sym.cpp \
../PostHartreeFock/TestOperators.cpp \
../PostHartreeFock/TestOperators2.cpp \
../PostHartreeFock/als_ev.cpp \
../PostHartreeFock/als_ev2.cpp \
../PostHartreeFock/als_ev_mo.cpp \
../PostHartreeFock/mals_ev.cpp 

CC_DEPS += \
./PostHartreeFock/Create1ParticleOperator.d \
./PostHartreeFock/Create2ParticleOperator.d 

OBJS += \
./PostHartreeFock/BasisTransform.o \
./PostHartreeFock/BuildTTOperator.o \
./PostHartreeFock/BuildTTOperator_molecular.o \
./PostHartreeFock/BuildTTOperator_parallel.o \
./PostHartreeFock/Create1ParticleOperator.o \
./PostHartreeFock/Create2ParticleOperator.o \
./PostHartreeFock/TestBuildTTOperator_with_sym.o \
./PostHartreeFock/TestOperators.o \
./PostHartreeFock/TestOperators2.o \
./PostHartreeFock/als_ev.o \
./PostHartreeFock/als_ev2.o \
./PostHartreeFock/als_ev_mo.o \
./PostHartreeFock/mals_ev.o 

CPP_DEPS += \
./PostHartreeFock/BasisTransform.d \
./PostHartreeFock/BuildTTOperator.d \
./PostHartreeFock/BuildTTOperator_molecular.d \
./PostHartreeFock/BuildTTOperator_parallel.d \
./PostHartreeFock/TestBuildTTOperator_with_sym.d \
./PostHartreeFock/TestOperators.d \
./PostHartreeFock/TestOperators2.d \
./PostHartreeFock/als_ev.d \
./PostHartreeFock/als_ev2.d \
./PostHartreeFock/als_ev_mo.d \
./PostHartreeFock/mals_ev.d 


# Each subdirectory must supply rules for building sources it contributes
PostHartreeFock/%.o: ../PostHartreeFock/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

PostHartreeFock/%.o: ../PostHartreeFock/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


