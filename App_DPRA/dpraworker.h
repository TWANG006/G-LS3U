#ifndef DPRAWORKER_H
#define DPRAWORKER_H

#include <QObject>
#include "aia_cpuf.h"

class DPRAWorker : public QObject
{
	Q_OBJECT

public:
	DPRAWorker() = default;
	~DPRAWorker();

public slots:
	void computeAIA(const QStringList& aiaimgList,
					const std::vector<float>& deltas,
					const int iMaxIterations,
					const float fMaxErr,
					const int iNumThreads);	
	void computeDPRA(const QStringList& dpraimgList, 
					 const int iWidth,
					 const int iHeihgt,
					 const int iNumThreads);

signals:
	void imgSizeErr();

private:
	
	int m_iWidth;
	int m_iHeight;

	// AIA params
	std::vector<cv::Mat> m_AIAimgs;
	std::vector<float> m_deltas;
	std::vector<float> m_refPhi;
	AIA::AIA_CPU_DnF m_aia;

	// DPRA params
	std::unique_ptr<DPRA::DPRA_HYBRIDF> m_dpraPtr;
};

#endif // DPRAWORKER_H
