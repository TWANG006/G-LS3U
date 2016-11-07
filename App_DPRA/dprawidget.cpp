#include "stdafx.h"
#include "dprawidget.h"


DPRAWidget::DPRAWidget(QWidget *parent)
	: QWidget(parent)
	, m_iWidth(0)
	, m_iHeight(0)
	, m_dpraPtr(nullptr)
{
	ui.setupUi(this);
	ui.computeAIAgroupBox->hide();	

	connect(ui.loadphiButton, SIGNAL(clicked()), this, SLOT(openPhi()));
	connect(ui.loadAIAImgButton, SIGNAL(clicked()), this, SLOT(openAIAImages()));
	connect(ui.computeAIAButton, SIGNAL(clicked()), this, SLOT(computeAIA()));
	connect(ui.exitButton, SIGNAL(clicked()), this, SIGNAL(onexit()));
	connect(ui.computeDPRAButton, SIGNAL(clicked()), this, SLOT(computeDPRA()));
}

DPRAWidget::~DPRAWidget()
{
	//if (m_dpraworkerThread->isRunning())
	//{
	//	m_dpraworkerThread->quit();
	//	m_dpraworkerThread->wait();
	//}
}

void DPRAWidget::openPhi()
{
	// Open Phi & remember the last opened path
	m_phiFileName = QFileDialog::getOpenFileName(
		this, 
		tr("Open Ref Phi Files"), 
		m_filePath, 
		tr("Phi (*.bin)"));

	if (!m_phiFileName.isNull())
	{
		QFileInfo fileInfo(m_phiFileName);
		m_filePath = fileInfo.path();

		// Load the phi&save it to a QVector
		float *refPhi = nullptr;
		WFT_FPA::Utils::ReadMatrixFromDisk(m_phiFileName.toStdString().c_str(), &m_iHeight, &m_iWidth, &refPhi);
		
		mv_refPhi.resize(m_iWidth * m_iHeight, 0);
		memcpy(mv_refPhi.data(), refPhi, sizeof(float)*m_iWidth*m_iHeight);

		free(refPhi); refPhi = nullptr;

		// Enable the DPRA group box since we have the initial reference phi
		ui.DPRABox->setEnabled(true);
	}
}

void DPRAWidget::openAIAImages()
{
	// Open AIA images & remember the opened path
	m_AIAImgFileList = QFileDialog::getOpenFileNames(
		this, 
		tr("Open AIA images"), 
		m_filePath, 
		tr("Img (*.png *.bmp *.jpg *.tif *.jpeg)"));

	if(!m_AIAImgFileList.isEmpty())
	{
		QFileInfo fileInfor(m_AIAImgFileList.at(0));
		m_filePath = fileInfor.path();
	}
}

void DPRAWidget::computeAIA()
{
	if(m_AIAImgFileList.isEmpty())
	{
		QMessageBox::critical(this, tr("Error!"), tr("No AIA images were selected!"));
		return;
	}

	// Get the size of the first image
	cv::Mat img = cv::imread(m_AIAImgFileList.at(0).toStdString());
	m_iWidth = img.cols;
	m_iHeight = img.rows;
	std::vector<cv::Mat> aiaimgs;

	foreach(const QString& imgstr, m_AIAImgFileList)
	{
		img = cv::imread(imgstr.toStdString());
		
		if (img.cols != m_iWidth || img.rows != m_iHeight)
		{
			QMessageBox::critical(this, tr("Error!"), tr("AIA images must have the same size!"));
			return;
		}

		// Convert&save the image to vector
		cv::cvtColor(img, img, CV_BGRA2GRAY);
		aiaimgs.push_back(img);
	}

	// Load the paramters
	m_iMaxIterations = ui.itersEdit->text().toInt();
	m_iNumberThreads = ui.threadsAIAspinBox->text().toInt();
	m_fMaxError = ui.errEdit->text().toFloat();

	// Load the initial delta phi's from the delta text edit
	QString textEdit = ui.textEdit->toPlainText();
	QTextStream in(&textEdit);

	std::vector<float> delta_phi;
	while(!in.atEnd())
	{
		delta_phi.push_back(in.readLine().toFloat());
	}


	/* Begin AIA computation */
	double time = 0;
	int iter = 0;
	float err = 0;
	// Make sure whether the number of delta phi's = number of images
	if (delta_phi.size() != m_AIAImgFileList.size())
	{
		QMessageBox::StandardButton reply = QMessageBox::question(
			this, 
			tr("Warning!"), 
			tr("Unmatched number of deltas (%1) and number of images (%2). \n").arg(delta_phi.size()).arg(m_AIAImgFileList.size()) +
			tr("Would you want to use random deltas instead? "), 
			QMessageBox::Yes | QMessageBox::No, 
			QMessageBox::Yes);

		if(QMessageBox::Yes == reply)	
		{
			
			m_aia(mv_refPhi, delta_phi, time, iter, err, aiaimgs, m_iMaxIterations, m_fMaxError, m_iNumberThreads);
		}
		else
		{
			// Do nothing or re-input delta phi's 
			return;
		}
	}
	else
	{
		// Use the input delta phi's to do the computation
		m_aia(mv_refPhi, delta_phi, time, iter, err, aiaimgs, m_iMaxIterations, m_fMaxError, m_iNumberThreads);

	}

	qDebug() << "Running Time of AIA is: " << time;
	qDebug() << "Number of Iterations is: " << iter;
	qDebug() << "Maximum Error is: " << err;

	// Enable the DPRA group box
	ui.DPRABox->setEnabled(true);
}

void DPRAWidget::openDPRAImages()
{
	// Open DPRA images & remember the opened path
	m_DPRAImgFileList = QFileDialog::getOpenFileNames(
		this, 
		tr("Open DPRA images"), 
		m_filePath, 
		tr("Img (*.png *.bmp *.jpg *.tif *.jpeg)"));

	if(!m_DPRAImgFileList.isEmpty())
	{
		QFileInfo fileInfor(m_DPRAImgFileList.at(0));
		m_filePath = fileInfor.path();
	}
}

void DPRAWidget::computeDPRA()
{
	qDebug() << "Compute DPRA";

	ui.DPRABox->setDisabled(true);
}