#ifndef DPRAWIDGET_H
#define DPRAWIDGET_H

#include <QWidget>
#include "ui_dprawidget.h"
#include "aia_cpuf.h"
#include "dpra_hybridf.h"

class DPRAWidget : public QWidget
{
	Q_OBJECT

public:
	DPRAWidget(QWidget *parent = 0);
	~DPRAWidget();

signals:
	void outputFileNameChanged(const QString&);
	void onexit();

private slots:
	void openPhi();
	void openAIAImages();
	void computeAIA();

	void openDPRAImages();
	void computeDPRA();

	void outputVideo();

private:
	bool videoWritter(const QString& fileName);

private:
	QString m_outputVideoFileName;
	QString m_phiFileName;
	QString m_filePath;

	QStringList m_AIAImgFileList;
	QStringList m_DPRAImgFileList;

	Ui::DPRAWidget ui;

	// AIA parameters
	int m_iMaxIterations;
	float m_fMaxError;
	int m_iNumberThreads;
	std::vector<float> mv_refPhi;
	AIA::AIA_CPU_DnF m_aia;

	// DPRA parameters
	int m_iWidth;
	int m_iHeight;
	
	std::vector<std::vector<float>> m_deltaPhiSum;
	std::unique_ptr<DPRA::DPRA_HYBRIDF> m_dpraPtr;

	//QThread *m_dpraworkerThread;
	//DPRAWorker *m_dpraworkerPtr;
};

#endif // DPRAWIDGET_H
