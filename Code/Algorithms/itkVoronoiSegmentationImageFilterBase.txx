/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkVoronoiSegmentationImageFilterBase.txx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkVoronoiSegmentationImageFilterBase_txx
#define __itkVoronoiSegmentationImageFilterBase_txx

#include "itkImageRegionIteratorWithIndex.h"
#include "itkVoronoiDiagram2DGenerator.h"
#include <math.h>

namespace itk
{

/* Constructor: setting the default parameter values. */
template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::VoronoiSegmentationImageFilterBase()
{
  m_Size.Fill(0);
  m_MinRegion = 20;
  m_Steps = 0;
  m_LastStepSeeds = 0;
  m_NumberOfSeeds = 200;
  m_NumberOfSeedsToAdded = 0;
  m_MeanDeviation = 0.8;
  m_UseBackgroundInAPrior = false;
  m_OutputBoundary = false;
  m_InteractiveSegmentation = false;
  m_WorkingVD=VoronoiDiagram::New();
  m_VDGenerator=VoronoiDiagramGenerator::New();
}

/* Destructor. */
template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::~VoronoiSegmentationImageFilterBase()
{
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Number Of Seeds: " 
     << m_NumberOfSeeds << std::endl;

  os << indent << "Minimum Region for Split: " 
     << m_MinRegion << std::endl;

  os << indent << "Number Of Steps to Run: (0 means runs until no region to split) " 
     << m_Steps << std::endl;

  os << indent << "UseBackgroundInAPrior = " << m_UseBackgroundInAPrior << std::endl;
  os << indent << "OutputBoundary = " << m_OutputBoundary << std::endl;
  os << indent << "MeanDeviation = " << m_MeanDeviation << std::endl;
  os << indent << "LastStepSeeds = " << m_LastStepSeeds << std::endl;
  os << indent << "InteractiveSegmentation = " << m_InteractiveSegmentation << std::endl;
  os << indent << "NumberOfSeedsToAdded = " << m_NumberOfSeedsToAdded
     << std::endl;
  os << indent << "Size = " << m_Size << std::endl;
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::GetPixelIndexFromPolygon(PointTypeDeque vertlist, IndexList *PixelPool)
{
  IndexType idx;
  PointType currP;
  PointType leftP;
  PointType rightP;

  currP = vertlist.front();
  vertlist.pop_front();
  leftP = vertlist.front();
  while(currP[1] > leftP[1])
    {
    vertlist.push_back(currP);
    currP=vertlist.front();
    vertlist.pop_front();
    leftP=vertlist.front();
    }
  rightP=vertlist.back();
  while(currP[1] > rightP[1])
    {
    vertlist.push_front(currP);
    currP=vertlist.back();
    vertlist.pop_back();
    rightP=vertlist.back();
    }
  leftP=vertlist.front();
  PointTypeDeque tmpQ; 
  tmpQ.clear();
  if(leftP[0]>rightP[0])
    {
    while(!(vertlist.empty()))
      {
      tmpQ.push_back(vertlist.front());
      vertlist.pop_front(); 
      }
    while(!(tmpQ.empty()))
      {
      vertlist.push_front(tmpQ.front());
      tmpQ.pop_front();
      }
    }
  tmpQ.clear();
  leftP=vertlist.front();
  rightP=vertlist.back();

  double beginy=currP[1];
  int intbeginy=(int)vcl_ceil(beginy); 
  idx[1]=intbeginy;
  double leftendy=leftP[1];
  double rightendy=rightP[1];
  double beginx=currP[0];
  double endx=currP[0];
  double leftDx,rightDx;
  double offset;
  double leftheadx=beginx;
  double rightheadx=endx;
  double leftheady=beginy;
  double rightheady=beginy;

  double endy;
  bool RorL;
  int i;
  if(leftendy>rightendy)
    {
    RorL=1;
    endy=rightendy;
    }
  else
    {
    RorL=0;
    endy=leftendy;
    }
  if (leftP[1]==beginy)
    {
    leftDx = 1.0;
    }
  else
    {
    leftDx=(leftP[0]-beginx)/(leftP[1]-beginy);
    }

  if (rightP[1]==beginy)
    {
    rightDx = 1.0;
    }
  else
    {
    rightDx=(rightP[0]-endx)/(rightP[1]-beginy);
    }

  int intendy=(int)vcl_floor(endy);
  if(intbeginy>intendy)
    { //no scanline
    if(RorL)
      {
      beginy=rightP[1];
      }
    else
      {
      beginy=leftP[1];
      }
    }
  else if((intbeginy==intendy) && (intbeginy==0))
    { //only one scanline at 0;
    if(RorL)
      {
      endx=rightP[0];
      }
    else
      {
      beginx=leftP[0];
      }
    for(i=static_cast<int>(vcl_ceil(beginx));i<=static_cast<int>(vcl_floor(endx));i++)
      {
      idx[0]=i;
      (*PixelPool).push_back(idx);
      }
    idx[1]=idx[1]+1;
    }
  else
    { //normal case some scanlines
    offset = (double)intbeginy-beginy;
    endx += offset * rightDx;
    beginx += offset * leftDx;
    while(idx[1] <= intendy)
      {
      for(i=static_cast<int>(vcl_ceil(beginx)); i <= static_cast<int>(vcl_floor(endx)); i++)
        {
        idx[0] = i;
        (*PixelPool).push_back(idx);
        }
      endx += rightDx;
      beginx += leftDx;
      idx[1] = idx[1]+1;
      }
    beginy=endy;
    }

  int vsize = static_cast<int>( vertlist.size() );
  while(vsize>2)
    {
    vsize--;
    if(RorL)
      {
      vertlist.pop_back();
      currP=rightP;
      rightheadx=currP[0];
      rightheady=currP[1];
      endx=currP[0];
      beginx=leftheadx+leftDx*(beginy-leftheady); 
      rightP=vertlist.back();
      if ((rightP[1]==currP[1]))
        {
        rightDx = 1.0;
        }
      else
        {
        rightDx=(rightP[0]-currP[0])/(rightP[1]-currP[1]);
        }
      }
    else
      {
      vertlist.pop_front();
      currP=leftP;
      leftheadx=currP[0];
      leftheady=currP[1];
      beginx=currP[0];
      endx=rightheadx+rightDx*(beginy-rightheady);
      leftP=vertlist.front();
      if ((leftP[1]==currP[1]))
        {
        leftDx = 1.0;
        }
      else
        {
        leftDx=(leftP[0]-currP[0])/(leftP[1]-currP[1]);
        }
      }
        
    leftendy=leftP[1];
    rightendy=rightP[1];
    if(leftendy>rightendy)
      {
      RorL=1;
      endy=rightendy;
      }
    else
      {
      RorL=0;
      endy=leftendy;
      }

    intendy=(int)vcl_floor(endy);
    intbeginy=(int)vcl_ceil(beginy); 

    if(intbeginy>intendy)
      { //no scanline
      if(RorL)
        {
        beginy=rightP[1];
        }
      else
        {
        beginy=leftP[1];
        }
      }
    else
      { //normal case some scanlines
      offset = (double)intbeginy-beginy;
      endx += offset * rightDx;
      beginx += offset * leftDx;
      while(idx[1] <= intendy)
        {
        for(i = static_cast<int>(vcl_ceil(beginx)); i <= static_cast<int>(vcl_floor(endx)); i++)
          {
          idx[0] = i;
          (*PixelPool).push_back(idx);
          }
        endx += rightDx;
        beginx += leftDx;
        idx[1] = idx[1] + 1;
        }
      beginy=idx[1];
      }
    }


  if(RorL)
    {
    beginy=rightP[1];
    endy=leftP[1];
    }
  else
    {
    beginy=leftP[1];
    endy=rightP[1];
    }
  intbeginy=(int)vcl_ceil(beginy);
  intendy=(int)vcl_floor(endy);
  if(intbeginy<=intendy)
    {
    if(RorL)
      {
      if ((rightP[1]==leftP[1]))
        {
        rightDx = 1.0;
        }
      else
        {
        rightDx=(rightP[0]-leftP[0])/(rightP[1]-leftP[1]);
        }
      endx=rightP[0];
      beginx=leftP[0]+leftDx*(rightP[1]-leftP[1]);
      }
    else
      {
      if ((rightP[1]==leftP[1]))
        {
        leftDx = 1.0;
        }
      else
        {
        leftDx=(rightP[0]-leftP[0])/(rightP[1]-leftP[1]);
        }
      beginx=leftP[0];
      endx=rightP[0]+rightDx*(leftP[1]-rightP[1]);
      }
    offset = (double)intbeginy-beginy;
    beginx += offset * leftDx;
    endx += offset * rightDx;
    while(idx[1] <= intendy)
      {
      for(i = static_cast<int>(vcl_ceil(beginx)); i <= static_cast<int>(vcl_floor(endx)); i++)
        {
        idx[0] = i;
        (*PixelPool).push_back(idx);
        }
      endx += rightDx;
      beginx += leftDx;
      idx[1] = idx[1] + 1;
      }
    }
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::ClassifyDiagram(void)
{
  PointIdIterator currPit;
  PointIdIterator currPitEnd;
  PointType currP;
  PointTypeDeque VertList;
  IndexList PixelPool;
  for(int i=0;i<m_NumberOfSeeds;i++)
    {
    CellAutoPointer currCell;
    m_WorkingVD->GetCellId(i,currCell);
    currPitEnd = currCell->PointIdsEnd();
    VertList.clear();
    for(currPit = currCell->PointIdsBegin(); currPit != currPitEnd; ++currPit)
      {
      m_WorkingVD->GetPoint((*currPit),&(currP));
      VertList.push_back(currP);
      }

    PixelPool.clear();
    this->GetPixelIndexFromPolygon(VertList,&PixelPool);
    m_NumberOfPixels[i] = static_cast<int>( PixelPool.size() );
    m_Label[i] = this->TestHomogeneity(PixelPool);
    }

  m_NumberOfBoundary = 0;
  for(int i=0;i<m_NumberOfSeeds;i++)
    {
    // if not homogeneous
    if(m_Label[i] == 0)
      {
      NeighborIdIterator itend = m_WorkingVD->NeighborIdsEnd(i);
      NeighborIdIterator it=m_WorkingVD->NeighborIdsBegin(i);
      bool bnd = 0;
      // and any adjacent region is homogeneous
      while((it != itend) && (!bnd))
        {
        bnd = (m_Label[*it] == 1);
        ++it;
        }
      // then this must be a boundary region
      if(bnd)
        {
        m_Label[i] = 2;
        m_NumberOfBoundary++;
        }
      }
    }
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::GenerateAddingSeeds(void)
{
  EdgeIterator eit;
  EdgeIterator eitend = m_WorkingVD->EdgeEnd();  
  PointType adds;
  Point<int,2> seeds;

  for(eit = m_WorkingVD->EdgeBegin();eit != eitend; ++eit)
    {
    seeds = m_WorkingVD->GetSeedsIDAroundEdge(&*eit);
    if( ((m_Label[seeds[0]]==2)||(m_Label[seeds[1]]==2))
        && (m_NumberOfPixels[seeds[0]]>m_MinRegion)
        && (m_NumberOfPixels[seeds[1]]>m_MinRegion) )
      {
      adds[0] = (eit->m_Left[0] + eit->m_Right[0])*0.5;
      adds[1] = (eit->m_Left[1] + eit->m_Right[1])*0.5;
      m_SeedsToAdded.push_back(adds);
      }
    }
}


template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::RunSegmentOneStep(void)
{
  m_NumberOfPixels.resize(m_NumberOfSeeds);
  m_Label.resize(m_NumberOfSeeds);
  m_SeedsToAdded.clear();
  m_VDGenerator->Update();
  m_WorkingVD=m_VDGenerator->GetOutput();
  this->ClassifyDiagram();
  this->GenerateAddingSeeds();
  m_NumberOfSeedsToAdded = static_cast<int>( m_SeedsToAdded.size() );

  if (m_InteractiveSegmentation)
    {
    if(m_OutputBoundary)
      {
      this->MakeSegmentBoundary();
      }
    else
      {
      this->MakeSegmentObject();
      }
    }
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::RunSegment(void)
{
  bool ok = 1;
  if(m_Steps == 0)
    {
    unsigned long count=1;
    this->RunSegmentOneStep();
    // guess at a progress
    this->UpdateProgress(static_cast<float>(count)
                         / static_cast<float>(NumericTraits<unsigned long>::max()));
    if(m_NumberOfBoundary == 0)
      {
      ok=0;
      }
    while( (m_NumberOfSeedsToAdded != 0) && ok)
      {
      count++;
      m_VDGenerator->AddSeeds(m_NumberOfSeedsToAdded,m_SeedsToAdded.begin());
      m_LastStepSeeds = m_NumberOfSeeds;
      m_NumberOfSeeds += m_NumberOfSeedsToAdded;
      this->RunSegmentOneStep();
      // guess at a progress
      this->UpdateProgress(static_cast<float>(count)
                           / static_cast<float>(NumericTraits<unsigned long>::max()));
      }
    }
  else if(m_Steps == 1)
    {
    this->RunSegmentOneStep();
    this->UpdateProgress(1.0);
    }
  else
    {
    this->RunSegmentOneStep();
    if (m_Steps == 0)
      {
      this->UpdateProgress(1.0);
      }
    else
      {
      this->UpdateProgress(1.0/static_cast<float>(m_Steps));
      }
    if(m_NumberOfBoundary == 0)
      {
      ok=0;
      }
    int i = 1;
    while((i<m_Steps) && ok)
      {
      m_VDGenerator->AddSeeds(m_NumberOfSeedsToAdded, m_SeedsToAdded.begin());
      m_LastStepSeeds = m_NumberOfSeeds;
      m_NumberOfSeeds += m_NumberOfSeedsToAdded;
      this->RunSegmentOneStep();
      i++;
      this->UpdateProgress(static_cast<float>(i)/static_cast<float>(m_Steps));
      }
    }   
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::GenerateData(void)
{
  // Allocate the output
  this->GetOutput()
    ->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();

  // This ivar should be necessary
  m_Size = this->GetInput()->GetRequestedRegion().GetSize();

  VoronoiDiagram::PointType VDsize;
  VDsize[0] = (VoronoiDiagram::CoordRepType)(m_Size[0]-0.1);
  VDsize[1] = (VoronoiDiagram::CoordRepType)(m_Size[1]-0.1);
  m_VDGenerator->SetBoundary(VDsize);
  m_VDGenerator->SetRandomSeeds(m_NumberOfSeeds);

  this->RunSegment();
  if(m_OutputBoundary)
    {
    this->MakeSegmentBoundary();
    }
  else
    {
    this->MakeSegmentObject();
    }

}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::BeforeNextStep(void)
{ 
  m_VDGenerator->AddSeeds(m_NumberOfSeedsToAdded, m_SeedsToAdded.begin()); 
  m_LastStepSeeds = m_NumberOfSeeds; 
  m_NumberOfSeeds += m_NumberOfSeedsToAdded; 
} 


template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::MakeSegmentBoundary(void)
{

  RegionType region = this->GetInput()->GetRequestedRegion(); 
  itk::ImageRegionIteratorWithIndex <OutputImageType> oit(this->GetOutput(), region); 
  while( !oit.IsAtEnd())
    {
    oit.Set(0); 
    ++oit; 
    }

  NeighborIdIterator nit;
  NeighborIdIterator nitend;
  for(int i=0;i<m_NumberOfSeeds;i++)
    {
    if(m_Label[i] == 2)
      {
      nitend = m_WorkingVD->NeighborIdsEnd(i);
      for(nit=m_WorkingVD->NeighborIdsBegin(i); nit != nitend; ++nit)
        {
        if(((*nit)>i)&&(m_Label[*nit]==2))
          {
          drawLine(m_WorkingVD->GetSeed(i),m_WorkingVD->GetSeed(*nit));
          i=i;
          }
        }
      }
    }
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::MakeSegmentObject(void)
{
  RegionType region = this->GetInput()->GetRequestedRegion(); 
  itk::ImageRegionIteratorWithIndex <OutputImageType> oit(this->GetOutput(), region); 
  while( !oit.IsAtEnd())
    {
    oit.Set(0); 
    ++oit; 
    }
  PointIdIterator currPit;
  PointIdIterator currPitEnd;
  PointType currP;
  PointTypeDeque VertList;
  for(int i=0;i<m_NumberOfSeeds;i++)
    {
    if(m_Label[i] == 1)
      {
      CellAutoPointer currCell; 
      m_WorkingVD->GetCellId(i, currCell);
      currPitEnd = currCell->PointIdsEnd();
      VertList.clear();
      for(currPit = currCell->PointIdsBegin(); currPit != currPitEnd; ++currPit)
        {
        m_WorkingVD->GetPoint((*currPit),&(currP));
        VertList.push_back(currP);
        }
      FillPolygon(VertList);
      }
    }
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::FillPolygon(PointTypeDeque vertlist, OutputPixelType color)
{
  IndexType idx;

  PointType currP;
  PointType leftP;
  PointType rightP;
  currP = vertlist.front();
  vertlist.pop_front();
  leftP = vertlist.front();
  while(currP[1] > leftP[1])
    {
    vertlist.push_back(currP);
    currP=vertlist.front();
    vertlist.pop_front();
    leftP=vertlist.front();
    }
  rightP=vertlist.back();
  while(currP[1] > rightP[1])
    {
    vertlist.push_front(currP);
    currP=vertlist.back();
    vertlist.pop_back();
    rightP=vertlist.back();
    }
  leftP=vertlist.front();
  PointTypeDeque tmpQ; 
  tmpQ.clear();
  if(leftP[0]>rightP[0])
    {
    while(!(vertlist.empty()))
      {
      tmpQ.push_back(vertlist.front());
      vertlist.pop_front(); 
      }
    while(!(tmpQ.empty()))
      {
      vertlist.push_front(tmpQ.front());
      tmpQ.pop_front();
      }
    } 
  tmpQ.clear();
  leftP=vertlist.front();
  rightP=vertlist.back();
  
  double beginy=currP[1];
  int intbeginy=(int)vcl_ceil(beginy); 
  idx[1]=intbeginy;
  double leftendy=leftP[1];
  double rightendy=rightP[1];
  double beginx=currP[0];
  double endx=currP[0];
  double leftDx,rightDx;
  double offset;
  double leftheadx=beginx;
  double rightheadx=endx;
  double leftheady=beginy;
  double rightheady=beginy;

  double endy;
  bool RorL;
  int i;
  if(leftendy>rightendy)
    {
    RorL=1;
    endy=rightendy;
    }
  else
    {
    RorL=0;
    endy=leftendy;
    }
  if(leftP[1]==beginy)
    {
    leftDx = 1.0;
    }
  else
    {
    leftDx=(leftP[0]-beginx)/(leftP[1]-beginy);
    }
  if(rightP[1]==beginy)
    {
    rightDx = 1.0;
    }
  else
    {
    rightDx=(rightP[0]-endx)/(rightP[1]-beginy);
    }
  int intendy=(int)vcl_floor(endy);
  if(intbeginy>intendy)
    { //no scanline
    if(RorL)
      {
      beginy=rightP[1];
      }
    else
      {
      beginy=leftP[1];
      }
    }
  else if((intbeginy==intendy) && (intbeginy==0))
    { //only one scanline at 0;
    if(RorL)
      {
      endx=rightP[0];
      }
    else
      {
      beginx=leftP[0];
      }
    for(i=static_cast<int>(vcl_ceil(beginx));i<=static_cast<int>(vcl_floor(endx));i++)
      {
      idx[0]=i;
      this->GetOutput()->SetPixel(idx,color);  
      }
    idx[1] = idx[1] + 1;
    }
  else
    { //normal case some scanlines
    offset = (double)intbeginy-beginy;
    endx += offset * rightDx;
    beginx += offset * leftDx;
    while(idx[1] <= intendy)
      {
      for(i = static_cast<int>(vcl_ceil(beginx)); i <= static_cast<int>(vcl_floor(endx)); i++)
        {
        idx[0] = i;
        this->GetOutput()->SetPixel(idx,color);  
        }
      endx += rightDx;
      beginx += leftDx;
      idx[1] = idx[1] + 1;
      }
    beginy=endy;
    }

  int vsize = static_cast<int>( vertlist.size() );
  while(vsize>2)
    {
    vsize--;
    if(RorL)
      {
      vertlist.pop_back();
      currP=rightP;
      rightheadx=currP[0];
      rightheady=currP[1];
      endx=currP[0];
      beginx=leftheadx+leftDx*(beginy-leftheady); 
      rightP=vertlist.back(); 
      if (rightP[1] == currP[1])
        {
        rightDx = 1.0;
        }
      else
        {
        rightDx=(rightP[0]-currP[0])/(rightP[1]-currP[1]);
        }
      }
    else
      {
      vertlist.pop_front();
      currP=leftP;
      leftheadx=currP[0];
      leftheady=currP[1];
      beginx=currP[0];
      endx=rightheadx+rightDx*(beginy-rightheady);
      leftP=vertlist.front();
      if (leftP[1] == currP[1])
        {
        leftDx = 1.0;
        }
      else
        {
        leftDx=(leftP[0]-currP[0])/(leftP[1]-currP[1]);
        }
      }
        
    leftendy=leftP[1];
    rightendy=rightP[1];
    if(leftendy>rightendy)
      {
      RorL=1;
      endy=rightendy;
      }
    else
      {
      RorL=0;
      endy=leftendy;
      }

    intendy=(int)vcl_floor(endy);
    intbeginy=(int)vcl_ceil(beginy); 

    if(intbeginy>intendy)
      { //no scanline
      if(RorL)
        {
        beginy=rightP[1];
        }
      else
        {
        beginy=leftP[1];
        }
      }
    else
      { //normal case some scanlines
      offset = (double)intbeginy-beginy;
      endx += offset * rightDx;
      beginx += offset * leftDx;
      while(idx[1] <= intendy)
        {
        for(i = static_cast<int>(vcl_ceil(beginx)); i <= static_cast<int>(vcl_floor(endx)); i++)
          {
          idx[0] = i;
          this->GetOutput()->SetPixel(idx,color);  
          }
        endx += rightDx;
        beginx += leftDx;
        idx[1] = idx[1] + 1;
        }
      beginy = idx[1];
      }
    }
  

  if(RorL)
    {
    beginy=rightP[1];
    endy=leftP[1];
    }
  else
    {
    beginy=leftP[1];
    endy=rightP[1];
    }
  intbeginy=(int)vcl_ceil(beginy);
  intendy=(int)vcl_floor(endy);
  if(intbeginy<=intendy)
    {
    if(RorL)
      {
      if (rightP[1] == leftP[1])
        {
        rightDx = 1.0;
        }
      else
        {
        rightDx=(rightP[0]-leftP[0])/(rightP[1]-leftP[1]);
        }
      endx=rightP[0];
      beginx=leftP[0]+leftDx*(rightP[1]-leftP[1]);
      }
    else
      {
      if (rightP[1] == leftP[1])
        {
        leftDx = 1.0;
        }
      else
        {
        leftDx=(rightP[0]-leftP[0])/(rightP[1]-leftP[1]);
        }
      beginx=leftP[0];
      endx=rightP[0]+rightDx*(leftP[1]-rightP[1]);
      }
    offset = (double)intbeginy-beginy;
    beginx += offset * leftDx;
    endx += offset * rightDx;
    while(idx[1] <= intendy)
      {
      for(i = static_cast<int>(vcl_ceil(beginx)); i <= static_cast<int>(vcl_floor(endx)); i++)
        {
        idx[0] = i;
        this->GetOutput()->SetPixel(idx,color);  
        }
      endx += rightDx;
      beginx += leftDx;
      idx[1] = idx[1] + 1;
      }
    }
  
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage>
void
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::drawLine(PointType p1,PointType p2)
{
  int x1=(int)(p1[0]+0.5);
  int x2=(int)(p2[0]+0.5);
  int y1=(int)(p1[1]+0.5);
  int y2=(int)(p2[1]+0.5);
  if(x1==static_cast<int>(m_Size[0])) x1--;  
  if(x2==static_cast<int>(m_Size[0])) x2--;  
  if(y1==static_cast<int>(m_Size[1])) y1--;  
  if(y2==static_cast<int>(m_Size[1])) y2--;  
  
  
  int dx=x1-x2;
  int adx=(dx>0)?dx:-dx;
  int dy=y1-y2;
  int ady=(dy>0)?dy:-dy;
  int save;
  float curr;
  IndexType idx;
  if (adx > ady)
    {
    if(x1>x2)
      {
      save=x1; x1=x2; x2=save;
      y1=y2;
      }
    curr=(float)y1;
    if (dx == 0)
      {
      dx = 1;
      }
    float offset=(float)dy/dx;
    for(int i=x1;i<=x2;i++)
      {
      idx[0]=i;
      idx[1]=y1;
      this->GetOutput()->SetPixel(idx,1);
      curr += offset;
      y1=(int)(curr+0.5);
      }
    }
  else
    { 
    if(y1>y2)
      {
      x1=x2;
      save=y1; y1=y2; y2=save;
      }
    curr=(float)x1;
    if (dy == 0)
      {
      dy = 1;
      }
    float offset=(float)dx/dy;
    for(int i=y1;i<=y2;i++)
      {
      idx[0]=x1;
      idx[1]=i;
      this->GetOutput()->SetPixel(idx,1);
      curr += offset;
      x1=(int)(curr+0.5);
      }
    }
  
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage> 
void 
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::DrawDiagram(VDImagePointer result,unsigned char incolor, 
              unsigned char outcolor,unsigned char boundcolor) 
{ 
  
  RegionType region = this->GetInput()->GetRequestedRegion(); 
  itk::ImageRegionIteratorWithIndex <VDImage> vdit(result, region); 
  while( !vdit.IsAtEnd())
    {
    vdit.Set(0); 
    ++vdit; 
    } 
  
  EdgeIterator eit; 
  EdgeIterator eitend = m_WorkingVD->EdgeEnd();   
  PointType adds; 
  Point<int,2> seeds; 
  for(eit = m_WorkingVD->EdgeBegin();eit != eitend; ++eit)
    { 
    seeds = m_WorkingVD->GetSeedsIDAroundEdge(&*eit); 
    if((m_Label[seeds[0]]==2)||(m_Label[seeds[1]]==2))
      { 
      drawVDline(result,eit->m_Left,eit->m_Right,boundcolor); 
      } 
    else if(m_Label[seeds[0]])
      { 
      drawVDline(result,eit->m_Left,eit->m_Right,incolor); 
      } 
    else
      { 
      drawVDline(result,eit->m_Left,eit->m_Right,outcolor); 
      } 
    } 
  
} 
    
template <class TInputImage, class TOutputImage, class TBinaryPriorImage> 
void 
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::drawVDline(VDImagePointer result,PointType p1,PointType p2,
             unsigned char color) 
{ 
  int x1=(int)(p1[0]+0.5); 
  int x2=(int)(p2[0]+0.5); 
  int y1=(int)(p1[1]+0.5); 
  int y2=(int)(p2[1]+0.5); 
  if(x1==(int)m_Size[0])
    {
    x1--;
    }
  if(x2==(int)m_Size[0])
    {
    x2--;
    }
  if(y1==(int)m_Size[1])
    {
    y1--;
    }
  if(y2==(int)m_Size[1])
    {
    y2--;
    }
  int dx=x1-x2; 
  int adx=(dx>0)?dx:-dx; 
  int dy=y1-y2; 
  int ady=(dy>0)?dy:-dy; 
  int save; 
  float curr; 
  IndexType idx; 
  if (adx > ady)
    { 
    if(x1>x2)
      { 
      save=x1; x1=x2; x2=save; 
      save=y1; y1=y2; y2=save; 
      } 
    curr=(float)y1; 
    if (dx == 0)
      {
      dx = 1;
      }
    float offset=(float)dy/dx; 
    for(int i=x1;i<=x2;i++)
      { 
      idx[0]=i; 
      idx[1]=y1; 
      result->SetPixel(idx,color); 
      curr += offset; 
      y1=(int)(curr+0.5); 
      } 
    } 
  else
    {  
    if(y1>y2)
      { 
      save=x1; x1=x2; x2=save; 
      save=y1; y1=y2; y2=save; 
      } 
    curr=(float)x1; 
    if (dy == 0)
      {
      dy = 1;
      }
    float offset=(float)dx/dy; 
    for(int i=y1;i<=y2;i++)
      { 
      idx[0]=x1; 
      idx[1]=i; 
      result->SetPixel(idx,color); 
      curr += offset; 
      x1=(int)(curr+0.5); 
      } 
    } 
} 

template <class TInputImage, class TOutputImage, class TBinaryPriorImage> 
void 
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::GenerateInputRequestedRegion()
{
  // call the superclass's method
  Superclass::GenerateInputRequestedRegion();

  // set the input requested region to the LargestPossibleRegion
    {
    InputImagePointer input =
      const_cast< InputImageType * >( this->GetInput() );
    if (input)
      {
      input->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
      }
    }
}

template <class TInputImage, class TOutputImage, class TBinaryPriorImage> 
void 
VoronoiSegmentationImageFilterBase <TInputImage,TOutputImage,TBinaryPriorImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  // call the superclass's method
  Superclass::EnlargeOutputRequestedRegion(output);

  // set the output requested region to the LargestPossibleRegion
  this->GetOutput()
    ->SetRequestedRegion( this->GetOutput()->GetLargestPossibleRegion() );
}

} //end namespace

#endif