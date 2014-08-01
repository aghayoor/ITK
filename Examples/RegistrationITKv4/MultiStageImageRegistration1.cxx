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

//  Software Guide : BeginCommandLineArgs
//    INPUTS:  {BrainT1SliceBorder20.png}
//    INPUTS:  {BrainProtonDensitySliceShifted13x17y.png}
//    OUTPUTS: {MultiResImageRegistration2Output.png}
//    ARGUMENTS:    100
//    OUTPUTS: {MultiResImageRegistration2CheckerboardBefore.png}
//    OUTPUTS: {MultiResImageRegistration2CheckerboardAfter.png}
//  Software Guide : EndCommandLineArgs

// Software Guide : BeginLatex
//
//  This example illustrates the use of more complex components of the
//  registration framework. In particular, it introduces a multistage,
//  multi-resolutionary approach to run a registration process using two
//  linear \doxygen{TranslationTransform} and \doxygen{AffineTransform}.
//  Also, it shows the use of emph{Scale Estimators}
//  for fine-tuning the scale parameters of the optimizer when an Affine
//  transform is used. The \doxygen{RegistrationParameterScalesFromPhysicalShift}
//  filter is used for automatic estimation of parameters scales.
//
// \index{itk::ImageRegistrationMethod!AffineTransform}
// \index{itk::ImageRegistrationMethod!Scaling parameter space}
// \index{itk::AffineTransform!Image Registration}
// \index{itk::ImageRegistrationMethodv4!Multi-Stage}
// \index{itk::ImageRegistrationMethodv4!Multi-Resolution}
// \index{itk::ImageRegistrationMethodv4!Multi-Modality}
//
// To begin the example, we include the headers of the registration
// components we will use.
//
// Software Guide : EndLatex

// Software Guide : BeginCodeSnippet
#include "itkImageRegistrationMethodv4.h"

#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkJointHistogramMutualInformationImageToImageMetricv4.h"

#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"

#include "itkTranslationTransform.h"
#include "itkAffineTransform.h"
#include "itkCompositeTransform.h"
// Software Guide : EndCodeSnippet

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkCheckerBoardImageFilter.h"

#include "itkCommand.h"

//  The following section of code implements a Command observer
//  that will monitor the configurations of the registration process
//  at every change of stage and resolution level.
//
template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
  typedef  RegistrationInterfaceCommand   Self;
  typedef  itk::Command                   Superclass;
  typedef  itk::SmartPointer<Self>        Pointer;
  itkNewMacro( Self );

protected:
  RegistrationInterfaceCommand() {};

public:
  typedef   TRegistration                          RegistrationType;
//  typedef   RegistrationType *                     RegistrationPointer;
//  typedef   itk::ObjectToObjectOptimizerBase       OptimizerType;
//  typedef   OptimizerType *                        OptimizerPointer;

  // The Execute function simply calls another version of the \code{Execute()}
  // method accepting a \code{const} input object
  void Execute( itk::Object * object, const itk::EventObject & event)
    {
    Execute( (const itk::Object *) object , event );
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    if( !(itk::MultiResolutionIterationEvent().CheckEvent( &event ) ) )
      {
      return;
      }

    std::cout << "Observing from class " << object->GetNameOfClass();

    const RegistrationType * registration =
                                dynamic_cast<const RegistrationType *>( object );
/*
    const OptimizerPointer optimizer =
                            dynamic_cast<const OptimizerPointer>( registration->GetOptimizer() );
*/
    unsigned int currentLevel = registration->GetCurrentLevel();
    typename RegistrationType::ShrinkFactorsPerDimensionContainerType shrinkFactors =
                                              registration->GetShrinkFactorsPerDimension( currentLevel );
    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmas =
                                                            registration->GetSmoothingSigmasPerLevel();

    std::cout << "-------------------------------------" << std::endl;
    std::cout << " Current multi-resolution level = " << currentLevel << std::endl;
    std::cout << "    shrink factor = " << shrinkFactors << std::endl;
    std::cout << "    smoothing sigma = " << smoothingSigmas[currentLevel] << std::endl;
    std::cout << std::endl;
    }
};

//  The following section of code implements an observer
//  that will monitor the evolution of the registration process.
//
class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate(): m_CumulativeIterationIndex(0) {};

public:
  typedef   itk::RegularStepGradientDescentOptimizerv4<double>  OptimizerType;
  typedef   const OptimizerType *                               OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    OptimizerPointer optimizer =
      dynamic_cast< OptimizerPointer >( object );
    if( !(itk::IterationEvent().CheckEvent( &event )) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << "  " <<
      m_CumulativeIterationIndex++ << std::endl;
    }

private:
  unsigned int m_CumulativeIterationIndex;
};

int main( int argc, char *argv[] )
{
  if( argc < 4 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << " outputImagefile [backgroundGrayLevel]";
    std::cerr << " [checkerboardbefore] [CheckerBoardAfter]";
    std::cerr << " [numberOfBins] " << std::endl;
    return EXIT_FAILURE;
    }

  const    unsigned int    Dimension = 2;
  typedef  float           PixelType;

  typedef itk::Image< PixelType, Dimension >  FixedImageType;
  typedef itk::Image< PixelType, Dimension >  MovingImageType;

  //  Software Guide : BeginLatex
  //
  //  In a multistage scenario, each stage needs an individual instantiation
  //  of the \doxygen{ImageRegistrationMethodv4}, so each stage can possibly
  //  have a different transform, a different optimizer, and a different image
  //  metric and can be performed in multiple levels.
  //  The configuration of the registration method at each stage closely
  //  follows the procedure in the previous section.
  //
  //  In early stages we can use simpler transforms with more aggressive parameters
  //  set to take big steps toward the optimal value. Then, at the final stage
  //  we can have a more complex transform to do fine adjustments of the final
  //  parameters.
  //
  //  A possible scheme is to use a simple translation transform for initial
  //  coarse registration levels and upgrade to an affine transform at the
  //  finer level.
  //  Since we have two different types of transforms, we can use a multistage
  //  registration approach as shown in the current example.
  //
  //  First we need to configure the registration components of the initial stage.
  //  The instantiation of the transform type requires only the
  //  dimension of the space and the type used for representing space coordinates.
  //
  //  \index{itk::TranslationTransform!Instantiation}
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::TranslationTransform< double, Dimension >              TransformType;
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  The types of other registration components are as previous examples.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::RegularStepGradientDescentOptimizerv4<double>   TranslationOptimizerType;

  typedef itk::MattesMutualInformationImageToImageMetricv4<
                                          FixedImageType,
                                          MovingImageType >    TranslationMetricType;

  typedef itk::ImageRegistrationMethodv4<
                                    FixedImageType,
                                    MovingImageType,
                                    TransformType    >         TranslationRegistrationType;
  // Software Guide : EndCodeSnippet

  //  All the components are instantiated using their \code{New()} method
  //  and connected to the registration object as in previous example.
  //
  TranslationOptimizerType::Pointer      transOptimizer     = TranslationOptimizerType::New();
  TranslationMetricType::Pointer         transMetric        = TranslationMetricType::New();
  TranslationRegistrationType::Pointer   transRegistration  = TranslationRegistrationType::New();

  transRegistration->SetOptimizer(     transOptimizer     );
  transRegistration->SetMetric( transMetric  );

  //  Software Guide : BeginLatex
  //
  //  The output transform will be constructed internally in the
  //  registration method since the \emph{TransformType} is passed
  //  to the registration filter as a template parameter.
  //  However, we should provide an initial moving transform for the
  //  registration method if needed.
  //
  //  \index{itk::TranslationTransform!New()}
  //  \index{itk::TranslationTransform!Pointer}
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  TransformType::Pointer   movingInitTransform  = TransformType::New();
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  Then, initial transform can be passed to the registration
  //  filter by \code{SetMovingInitialTransform()} method.
  //
  //  \index{itk::Image\-Registration\-Methodv4!SetMovingInitialTransform()}
  //
  //  Software Guide : EndLatex

  typedef TranslationOptimizerType::ParametersType ParametersType;
  ParametersType initialParameters( movingInitTransform->GetNumberOfParameters() );

  initialParameters[0] = 0.0;  // Initial offset in mm along X
  initialParameters[1] = 0.0;  // Initial offset in mm along Y

  movingInitTransform->SetParameters( initialParameters );

  // Software Guide : BeginCodeSnippet
  transRegistration->SetMovingInitialTransform( movingInitTransform );
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  We can use a \doxygen{CompositeTransform} to hold the final transform
  //  of the registration process resulted from multiple stages. This composite
  //  transform should also hold the moving initial transform (if it exists)
  //  because as explained in section \ref{sec:RigidRegistrationIn2D},
  //  the initial transform is not updated during the registration process
  //  while it is used for evaluation of the metric values.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::CompositeTransform< double,
                                   Dimension >  CompositeTransformType;
  CompositeTransformType::Pointer   outputCompTransform  =
                                                CompositeTransformType::New();
  outputCompTransform->AddTransform( movingInitTransform );
  // Software Guide : EndCodeSnippet

  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(  argv[1] );
  movingImageReader->SetFileName( argv[2] );

  transRegistration->SetFixedImage(    fixedImageReader->GetOutput()    );
  transRegistration->SetMovingImage(   movingImageReader->GetOutput()   );
  transRegistration->SetObjectName("TranslationRegistration");

  //  Software Guide : BeginLatex
  //
  //  In the case of this simple example, we run only one level of registraion
  //  at a coarse resolution level for the first stage.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  const unsigned int numberOfLevels1 = 1;

  TranslationRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel1;
  shrinkFactorsPerLevel1.SetSize( numberOfLevels1 );
  shrinkFactorsPerLevel1[0] = 3;

  TranslationRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel1;
  smoothingSigmasPerLevel1.SetSize( numberOfLevels1 );
  smoothingSigmasPerLevel1[0] = 2;

  transRegistration->SetNumberOfLevels ( numberOfLevels1 );
  transRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel1 );
  transRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel1 );
  // Software Guide : BeginCodeSnippet

  transMetric->SetNumberOfHistogramBins( 24 );

  transOptimizer->SetNumberOfIterations( 200 );
  transOptimizer->SetRelaxationFactor( 0.5 );

  //  Software Guide : BeginLatex
  //
  //  Also, we can use a more agressive paramter for the optimizer step size
  //  and more relaxed stop criteria.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  transOptimizer->SetLearningRate( 10 );
  transOptimizer->SetMinimumStepLength( 1.5 );
  // Software Guide : BeginCodeSnippet

  // Create the Command observer and register it with the optimizer.
  //
  CommandIterationUpdate::Pointer observer1 = CommandIterationUpdate::New();
  transOptimizer->AddObserver( itk::IterationEvent(), observer1 );

  // Create the Command interface observer and register it with the optimizer.
  //
  typedef RegistrationInterfaceCommand<TranslationRegistrationType> TranslationCommandType;
  TranslationCommandType::Pointer command1 = TranslationCommandType::New();
  transRegistration->AddObserver( itk::MultiResolutionIterationEvent(), command1 );

  //  Software Guide : BeginLatex
  //
  //  Once all the registration components are in place, we triger the registration
  //  process by calling \code{Update()} and add the result transform to the output
  //  composite transform. This composite transform can be used to initialize the next
  //  registration stage.
  //
  //  Software Guide : EndLatex

/*
  try
    {
    transRegistration->Update();
    std::cout << "Optimizer stop condition: "
              << transRegistration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return EXIT_FAILURE;
    }
*/

  // Software Guide : BeginCodeSnippet
  outputCompTransform->AddTransform( transRegistration->GetModifiableTransform() );
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  Now we can upgrade to an Affine transform as the second stage of registration
  //  process.
  //  The AffineTransform is a linear transformation that maps lines into
  //  lines. It can be used to represent translations, rotations, anisotropic
  //  scaling, shearing or any combination of them. Details about the affine
  //  transform can be seen in Section~\ref{sec:AffineTransform}.
  //  The instantiation of the transform type requires only the dimension of the
  //  space and the type used for representing space coordinates.
  //
  //  \index{itk::AffineTransform!Instantiation}
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::AffineTransform< double, Dimension > AffineTransformType;
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  We also use a different metric and optimizer in configuration of the second stage.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::ConjugateGradientLineSearchOptimizerv4Template<double>    AffineOptimizerType;

  typedef itk::JointHistogramMutualInformationImageToImageMetricv4<
                                                    FixedImageType,
                                                    MovingImageType >    AffineMetricType;

  typedef itk::ImageRegistrationMethodv4<
                                        FixedImageType,
                                        MovingImageType,
                                        AffineTransformType >            AffineRegistrationType;
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  All the components are instantiated using their \code{New()} method
  //  and connected to the registration object as in previous examples.
  //
  //  Software Guide : EndLatex

  AffineOptimizerType::Pointer      affineOptimizer     = AffineOptimizerType::New();
  AffineMetricType::Pointer         affineMetric        = AffineMetricType::New();
  AffineRegistrationType::Pointer   affineRegistration  = AffineRegistrationType::New();

  affineRegistration->SetOptimizer(     affineOptimizer     );
  affineRegistration->SetMetric( affineMetric  );

  //  Software Guide : BeginLatex
  //
  //  The current stage can be initialized using the initial moving transform
  //  and the result transform of the previous stage that both are concatenated
  //  into a composite transform.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  affineRegistration->SetMovingInitialTransform( outputCompTransform );
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  Metric parameters are set as follows.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  affineMetric->SetNumberOfHistogramBins( 20 );
  affineMetric->SetUseMovingImageGradientFilter( false );
  affineMetric->SetUseFixedImageGradientFilter( false );
  affineMetric->SetUseFixedSampledPointSet( false );
  affineMetric->SetVirtualDomainFromImage( fixedImageReader->GetOutput() );
  // Software Guide : EndCodeSnippet

  if( argc > 7 )
    {
    // optionally, override the values with numbers taken from the command line arguments.
    affineMetric->SetNumberOfHistogramBins( atoi( argv[7] ) );
    }

  // Set optimizer paramters
  //
  affineOptimizer->SetLowerLimit( 0 );
  affineOptimizer->SetUpperLimit( 2 );
  affineOptimizer->SetEpsilon( 0.2 );
  affineOptimizer->SetLearningRate( 4.0 );
  affineOptimizer->SetMaximumStepSizeInPhysicalUnits( 4.0 );
  affineOptimizer->SetNumberOfIterations( 200 );
  affineOptimizer->SetMinimumConvergenceValue( 0.01 );
  affineOptimizer->SetConvergenceWindowSize( 10 );

  affineRegistration->SetFixedImage( fixedImageReader->GetOutput() );
  affineRegistration->SetMovingImage( movingImageReader->GetOutput() );
  affineRegistration->SetObjectName("AffineRegistration");

  //  Software Guide : BeginLatex
  //
  //  The set of parameters in the AffineTransform have different
  //  dynamic ranges. Typically the parameters associated with the matrix
  //  have values around $[-1:1]$, although they are not restricted to this
  //  interval.  Parameters associated with translations, on the other hand,
  //  tend to have much higher values, typically in the order of $10.0$ to
  //  $100.0$. This difference in dynamic range negatively affects the
  //  performance of gradient descent optimizers. ITK provides some mechanisms to
  //  compensate for such differences in values among the parameters when
  //  they are passed to the optimizer.
  //
  //  The first mechanism consists of providing an
  //  array of scale factors to the optimizer. These factors re-normalize the
  //  gradient components before they are used to compute the step of the
  //  optimizer at the current iteration.
  //  These scales are estimated by the user intuitively as shown in previous
  //  examples of this chapter. In our particular case, a common choice
  //  for the scale parameters is to set to $1.0$ all those associated
  //  with the matrix coefficients, that is, the first $N \times N$
  //  factors. Then, we set the remaining scale factors to a small value.
  //
  //  Software Guide : EndLatex

  //  Software Guide : BeginLatex
  //
  //  Here the affine transform is represented by the matrix $\bf{M}$ and the
  //  vector $\bf{T}$. The transformation of a point $\bf{P}$ into $\bf{P'}$
  //  is expressed as
  //
  //  \begin{equation}
  //  \left[
  //  \begin{array}{c}
  //  {P'}_x  \\  {P'}_y  \\  \end{array}
  //  \right]
  //  =
  //  \left[
  //  \begin{array}{cc}
  //  M_{11} & M_{12} \\ M_{21} & M_{22} \\  \end{array}
  //  \right]
  //  \cdot
  //  \left[
  //  \begin{array}{c}
  //  P_x  \\ P_y  \\  \end{array}
  //  \right]
  //  +
  //  \left[
  //  \begin{array}{c}
  //  T_x  \\ T_y  \\  \end{array}
  //  \right]
  //  \end{equation}
  //
  //
  //  Software Guide : EndLatex

  //  Software Guide : BeginLatex
  //
  //  Based on the above discussion, we need very smaller scales for translation
  //  parameters of vector $\bf{T}$ ($T_x$, $T_y$) in compare with the parameters
  //  of matrix $\bf{M}$ ($M_{11}$, $M_{12}$, $M_{21}$, $M_{22}$).
  //  However, it is not that easy to have an intuitive estimation of all parameter
  //  scales when we have to deal with a large paramter space.
  //
  //  Fortunately, a framework for automated parameter scaling is provided by
  //  ITKv4. \doxygen{RegistrationParameterScalesEstimator} vastly reduce the
  //  difficulty of tuning parameters for different transform/metric combination.
  //  Parameter scales are estimated by analyzing the result of a small parameter
  //  update on the change in the magnitude of physical space deformation induced
  //  by the transformation.
  //
  //  The impact from a unit change of a parameter may be defined in multiple ways,
  //  such as the maximum shift of voxels in index or physical space, or the average
  //  norm of transform Jacobina.
  //  Filters \doxygen{RegistrationParameterScalesFromPhysicalShift}
  //  and \doxygen{RegistrationParameterScalesFromIndexShift} use the first definition
  //  to estimate the scales, while the \doxygen{RegistrationParameterScalesFromJacobian}
  //  filter estimates scales based on the later definition.
  //  In all methods, the goal is to rescale the transform parameters such that
  //  a unit change of each \emph{scaled parameter} will have the same impact on deformation.
  //
  //  In this example the first filter is chosen to estimate the paramter scales. Then
  //  scales estimator will be passed to optimizer.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  typedef itk::RegistrationParameterScalesFromPhysicalShift<
                                                            AffineMetricType>   ScalesEstimatorType;
  ScalesEstimatorType::Pointer scalesEstimator =
                                        ScalesEstimatorType::New();
  scalesEstimator->SetMetric( affineMetric );
  scalesEstimator->SetTransformForward( true );

  affineOptimizer->SetScalesEstimator( scalesEstimator );
  // Software Guide : EndCodeSnippet

  //  Software Guide : BeginLatex
  //
  //  The step length has to be proportional to the expected values of the
  //  parameters in the search space. Since the expected values of the matrix
  //  coefficients are around $1.0$, the initial step of the optimization
  //  should be a small number compared to $1.0$. As a guideline, it is
  //  useful to think of the matrix coefficients as combinations of
  //  $cos(\theta)$ and $sin(\theta)$.  This leads to use values close to the
  //  expected rotation measured in radians. For example, a rotation of $1.0$
  //  degree is about $0.017$ radians.
  //
  //  However, we do not need to be much woried about the above considerations
  //  Thanks to the \emph{ScalesEstimator}, the initial step size can also be
  //  estimated automatically, either at each iteration or only at the first
  //  iteration. Here, in this example we choose to estimate learning rate
  //  once at the begining of the registration process.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  affineOptimizer->SetDoEstimateLearningRateOnce( true );
  affineOptimizer->SetDoEstimateLearningRateAtEachIteration( false );
  // Software Guide : EndCodeSnippet

  // Create the Command observer and register it with the optimizer.
  //
  CommandIterationUpdate::Pointer observer2 = CommandIterationUpdate::New();
  affineOptimizer->AddObserver( itk::IterationEvent(), observer2 );

  //  Software Guide : BeginLatex
  //
  //  At the second stage, we run two levels of registration, where the second
  //  level is run in full resolution in which we do the final adjustments
  //  of the output parameters.
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  const unsigned int numberOfLevels2 = 2;

  AffineRegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel2;
  shrinkFactorsPerLevel2.SetSize( numberOfLevels2 );
  shrinkFactorsPerLevel2[0] = 2;
  shrinkFactorsPerLevel2[1] = 1;

  AffineRegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel2;
  smoothingSigmasPerLevel2.SetSize( numberOfLevels2 );
  smoothingSigmasPerLevel2[0] = 1;
  smoothingSigmasPerLevel2[0] = 0;

  affineRegistration->SetNumberOfLevels ( numberOfLevels2 );
  affineRegistration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel2 );
  affineRegistration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel2 );
  // Software Guide : EndCodeSnippet

  // Create the Command interface observer and register it with the optimizer.
  //
  typedef RegistrationInterfaceCommand<AffineRegistrationType> AffineCommandType;
  AffineCommandType::Pointer command2 = AffineCommandType::New();
  affineRegistration->AddObserver( itk::IterationEvent(), command2 );

  //  Software Guide : BeginLatex
  //
  //  Finally we triger the registration process by calling \code{Update()} once
  //  all the registration components are in place.
  //  Then, the result transform of the last stage is also added to the output
  //  composite transform that will be considered as the final transform of
  //  this multistage registration process and will be used by the resampler
  //  to resample the moving image in to the virtual domain space (fixed image space
  //  if there is no fixed initial transform).
  //
  //  Software Guide : EndLatex

  // Software Guide : BeginCodeSnippet
  try
    {
    affineRegistration->Update();
    std::cout << "Optimizer stop condition: "
              << affineRegistration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return EXIT_FAILURE;
    }

  outputCompTransform->AddTransform( affineRegistration->GetModifiableTransform() );
  // Software Guide : EndCodeSnippet

  std::cout << " Translation parameters after registration: " << std::endl
            << transOptimizer->GetCurrentPosition() << std::endl
            << " Last LearningRate: " << transOptimizer->GetCurrentStepLength() << std::endl;

  std::cout << " Affine parameters after registration: " << std::endl
            << affineOptimizer->GetCurrentPosition() << std::endl
            << " Last LearningRate: " << affineOptimizer->GetLearningRate() << std::endl;

///////////////////////////////
/*
  std::cout << "Optimizer Stopping Condition = "
            << optimizer->GetStopCondition() << std::endl;

  typedef RegistrationType::ParametersType ParametersType;
  ParametersType finalParameters = registration->GetLastTransformParameters();

  double TranslationAlongX = finalParameters[4];
  double TranslationAlongY = finalParameters[5];

  unsigned int numberOfIterations = optimizer->GetCurrentIteration();

  double bestValue = optimizer->GetValue();

  // Print out results
  //
  std::cout << "Result = " << std::endl;
  std::cout << " Translation X = " << TranslationAlongX  << std::endl;
  std::cout << " Translation Y = " << TranslationAlongY  << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;

*/
////////////////////////////

  //  Software Guide : BeginLatex
  //
  //  Let's execute this example using the same multi-modality images as
  //  before.  The registration converges after $5$ iterations in the first
  //  level, $7$ in the second level and $4$ in the third level. The final
  //  results when printed as an array of parameters are
  //
  //  \begin{verbatim}
  // [1.00164, 0.00147688, 0.00168372, 1.0027, 12.6296, 16.4768]
  //  \end{verbatim}
  //
  //  By reordering them as coefficient of matrix $\bf{M}$ and vector $\bf{T}$
  //  they can now be seen as
  //
  //  \begin{equation}
  //  M =
  //  \left[
  //  \begin{array}{cc}
  //  1.00164 & 0.0014 \\ 0.00168 & 1.0027 \\  \end{array}
  //  \right]
  //  \mbox{ and }
  //  T =
  //  \left[
  //  \begin{array}{c}
  //  12.6296  \\  16.4768  \\  \end{array}
  //  \right]
  //  \end{equation}
  //
  //  In this form, it is easier to interpret the effect of the
  //  transform. The matrix $\bf{M}$ is responsible for scaling, rotation and
  //  shearing while $\bf{T}$ is responsible for translations.  It can be seen
  //  that the translation values in this case closely match the true
  //  misalignment introduced in the moving image.
  //
  //  It is important to note that once the images are registered at a
  //  sub-pixel level, any further improvement of the registration relies
  //  heavily on the quality of the interpolator. It may then be reasonable to
  //  use a coarse and fast interpolator in the lower resolution levels and
  //  switch to a high-quality but slow interpolator in the final resolution
  //  level.
  //
  //  Software Guide : EndLatex

  typedef itk::ResampleImageFilter<
                            MovingImageType,
                            FixedImageType >    ResampleFilterType;
  ResampleFilterType::Pointer resample = ResampleFilterType::New();

  resample->SetTransform( outputCompTransform );
  resample->SetInput( movingImageReader->GetOutput() );

  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

  PixelType backgroundGrayLevel = 100;
  if( argc > 4 )
    {
    backgroundGrayLevel = atoi( argv[4] );
    }

  resample->SetSize(    fixedImage->GetLargestPossibleRegion().GetSize() );
  resample->SetOutputOrigin(  fixedImage->GetOrigin() );
  resample->SetOutputSpacing( fixedImage->GetSpacing() );
  resample->SetOutputDirection( fixedImage->GetDirection() );
  resample->SetDefaultPixelValue( backgroundGrayLevel );

  typedef  unsigned char                           OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CastImageFilter<
                        FixedImageType,
                        OutputImageType >          CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  WriterType::Pointer      writer =  WriterType::New();
  CastFilterType::Pointer  caster =  CastFilterType::New();

  writer->SetFileName( argv[3] );

  caster->SetInput( resample->GetOutput() );
  writer->SetInput( caster->GetOutput()   );
  writer->Update();

  //  Software Guide : BeginLatex
  //
  // \begin{figure}
  // \center
  // \includegraphics[width=0.32\textwidth]{MultiResImageRegistration2Output}
  // \includegraphics[width=0.32\textwidth]{MultiResImageRegistration2CheckerboardBefore}
  // \includegraphics[width=0.32\textwidth]{MultiResImageRegistration2CheckerboardAfter}
  // \itkcaption[Multi-Resolution Registration Input Images]{Mapped moving image
  // (left) and composition of fixed and moving images before (center) and
  // after (right) multi-resolution registration with the AffineTransform class.}
  // \label{fig:MultiResImageRegistration2Output}
  // \end{figure}
  //
  //  The result of resampling the moving image is shown in the left image
  //  of Figure \ref{fig:MultiResImageRegistration2Output}. The center and
  //  right images of the figure present a checkerboard composite of the fixed
  //  and moving images before and after registration.
  //
  //  Software Guide : EndLatex

  //  Software Guide : BeginLatex
  //
  // \begin{figure}
  // \center
  // \includegraphics[height=0.44\textwidth]{MultiResImageRegistration2TraceTranslations}
  // \includegraphics[height=0.44\textwidth]{MultiResImageRegistration2TraceMetric}
  // \itkcaption[Multi-Resolution Registration output plots]{Sequence of
  // translations and metric values at each iteration of the optimizer for
  // multi-resolution with the AffineTransform class.}
  // \label{fig:MultiResImageRegistration2Trace}
  // \end{figure}
  //
  //  Figure \ref{fig:MultiResImageRegistration2Trace} (left) presents the
  //  sequence of translations followed by the optimizer as it searched the
  //  parameter space. The right side of the same figure shows the sequence of
  //  metric values computed as the optimizer explored the parameter space.
  //
  //  Software Guide : EndLatex

  //
  // Generate checkerboards before and after registration
  //
  typedef itk::CheckerBoardImageFilter< FixedImageType > CheckerBoardFilterType;

  CheckerBoardFilterType::Pointer checker = CheckerBoardFilterType::New();

  checker->SetInput1( fixedImage );
  checker->SetInput2( resample->GetOutput() );

  caster->SetInput( checker->GetOutput() );
  writer->SetInput( caster->GetOutput()   );

  resample->SetDefaultPixelValue( 0 );

  // Write out checkerboard outputs
  // Before registration
  TransformType::Pointer identityTransform = TransformType::New();
  identityTransform->SetIdentity();
  resample->SetTransform( identityTransform );

  if( argc > 5 )
    {
    writer->SetFileName( argv[5] );
    writer->Update();
    }

  // After registration
  resample->SetTransform( outputCompTransform );
  if( argc > 6 )
    {
    writer->SetFileName( argv[6] );
    writer->Update();
    }

  return EXIT_SUCCESS;
}
