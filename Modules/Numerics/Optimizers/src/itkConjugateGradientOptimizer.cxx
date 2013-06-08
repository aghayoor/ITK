/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef _itkConjugateGradientOptimizer_hxx
#define _itkConjugateGradientOptimizer_hxx

#include "itkConjugateGradientOptimizer.h"

namespace itk
{
/**
 * Constructor
 */
ConjugateGradientOptimizer
::ConjugateGradientOptimizer()
{
  m_OptimizerInitialized    = false;
  m_VnlOptimizer            = 0;
}

/**
 * Destructor
 */
ConjugateGradientOptimizer
::~ConjugateGradientOptimizer()
{
  delete m_VnlOptimizer;
}

/**
 * Get the Optimizer
 */
vnl_conjugate_gradient *
ConjugateGradientOptimizer
::GetOptimizer(void)
{
  return m_VnlOptimizer;
}

/**
 * Connect a Cost Function
 */
void
ConjugateGradientOptimizer
::SetCostFunction(SingleValuedCostFunction *costFunction)
{
  const unsigned int numberOfParameters =
    costFunction->GetNumberOfParameters();

  CostFunctionAdaptorType *adaptor =
    new CostFunctionAdaptorType(numberOfParameters);

  adaptor->SetCostFunction(costFunction);

  if ( m_OptimizerInitialized )
    {
    delete m_VnlOptimizer;
    }

  this->SetCostFunctionAdaptor(adaptor);

  m_VnlOptimizer = new vnl_conjugate_gradient(*adaptor);
  m_OptimizerInitialized = true;
}

/** Return Current Value */
ConjugateGradientOptimizer::MeasureType
ConjugateGradientOptimizer
::GetValue() const
{
  const ParametersType & currentPositionInternalValue = this->GetCurrentPosition();
  const unsigned int paramSize=currentPositionInternalValue.size();
  InternalParametersType vnlCompatibleParameters(paramSize);

  const ScalesType & scales = this->GetScales();
  for ( unsigned int i = 0; i < paramSize; ++i )
    {
    vnlCompatibleParameters[i] = ( m_ScalesInitialized )
      ? currentPositionInternalValue[i] * scales[i]
      : currentPositionInternalValue[i];
    }
  return this->GetNonConstCostFunctionAdaptor()->f(vnlCompatibleParameters);
}

/**
 * Start the optimization
 */
void
ConjugateGradientOptimizer
::StartOptimization(void)
{
  this->InvokeEvent( StartEvent() );

  if ( this->GetMaximize() )
    {
    this->GetNonConstCostFunctionAdaptor()->NegateCostFunctionOn();
    }

  ParametersType currentPositionInternalValue( this->GetInitialPosition() );
  const unsigned int paramSize=currentPositionInternalValue.size();
  InternalParametersType vnlCompatibleParameters(paramSize);

  // We also scale the initial vnlCompatibleParameters up if scales are defined.
  // This compensates for later scaling them down in the cost function adaptor
  // and at the end of this function.
  const ScalesType & scales = this->GetScales();
  if ( m_ScalesInitialized )
    {
    this->GetNonConstCostFunctionAdaptor()->SetScales(scales);
    }
  for ( unsigned int i = 0; i < paramSize; ++i )
    {
    vnlCompatibleParameters[i] = ( m_ScalesInitialized )
      ? currentPositionInternalValue[i]*scales[i]
      : currentPositionInternalValue[i];
    }

  // vnl optimizers return the solution by reference
  // in the variable provided as initial position
  m_VnlOptimizer->minimize(vnlCompatibleParameters);

  // we scale the vnlCompatibleParameters down if scales are defined
  const ScalesType & invScales = this->GetInverseScales();
  for ( unsigned int i = 0; i < paramSize; ++i )
    {
    currentPositionInternalValue[i] = ( m_ScalesInitialized )
      ? vnlCompatibleParameters[i] * invScales[i]
      : vnlCompatibleParameters[i] ;
    }
  this->SetCurrentPosition(currentPositionInternalValue);
  this->InvokeEvent( EndEvent() );
}

/**
 * Get the maximum number of evaluations of the function.
 * In vnl this is used instead of a maximum number of iterations
 * given that an iteration could imply several evaluations.
 */
SizeValueType
ConjugateGradientOptimizer
::GetNumberOfIterations(void) const
{
  return m_VnlOptimizer->get_max_function_evals();
}

/**
 * Get the number of iterations in the last optimization.
 */
SizeValueType
ConjugateGradientOptimizer
::GetCurrentIteration(void) const
{
  return m_VnlOptimizer->get_num_iterations();
}
} // end namespace itk

#endif
