#include "stdafx.h"
#include "dpraworker.h"

//DPRAWorker::DPRAWorker(const int iMaxiterations, 
//					   const float fMaxErr, 
//					   const int iNumThreads,
//					   const std::vector<float>& deltas)
//	: m_dpraPtr(nullptr)
//	, m_deltas(deltas)
//	, m_iMaxiterations(iMaxiterations)
//	, m_fMaxErr(fMaxErr)
//	, m_iNumThreads(iNumThreads)
//	, m_aia()
//{}
//
//DPRAWorker::DPRAWorker(const std::vector<float>& refphi)
//	: m_dpraPtr(nullptr)
//	, m_refPhi(refphi)
//{}

void DPRAWorker::computeAIA(const QStringList& aiaimgList,
							const std::vector<float>& deltas,
							const int iMaxIterations,
							const float fMaxErr,
							const int iNumThreads)
{
	double time = 0;
	int iters = 0;
	float ferr = 0;

	// Get the size of the first image
	cv::Mat img = cv::imread(aiaimgList.at(0).toStdString());
	m_iWidth = img.cols;
	m_iHeight = img.rows;

	foreach(const QString& imgstr, aiaimgList)
	{
		img = cv::imread(imgstr.toStdString());
		
		if (img.cols != m_iWidth || img.rows != m_iHeight)
		{
			emit imgSizeErr();

			return;
		}

		// Convert&save the image to vector
		cv::cvtColor(img, img, CV_BGRA2GRAY);
		m_AIAimgs.push_back(img);
	}

	m_aia(m_refPhi, m_deltas, time, iters, ferr, m_AIAimgs, iMaxIterations, fMaxErr, iNumThreads);
	
	qDebug() << "Running Time of AIA is: " << time;
	qDebug() << "Number of Iterations is: " << iters;
	qDebug() << "Maximum Error is: " << ferr;
}

void DPRAWorker::computeDPRA(const QStringList& dpraimgList, 
							 const int iWidth, 
							 const int iHeight,
							 const int iNumThreads)
{

}

DPRAWorker::~DPRAWorker()
{

}
