################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../hartree-fock/hartree-fock++.cc \
../hartree-fock/hartree-fock.cc 

CC_DEPS += \
./hartree-fock/hartree-fock++.d \
./hartree-fock/hartree-fock.d 

OBJS += \
./hartree-fock/hartree-fock++.o \
./hartree-fock/hartree-fock.o 


# Each subdirectory must supply rules for building sources it contributes
hartree-fock/%.o: ../hartree-fock/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


