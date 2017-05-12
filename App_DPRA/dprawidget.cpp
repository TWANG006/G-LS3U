#include "stdafx.h"
#include "dprawidget.h"


DPRAWidget::DPRAWidget(QWidget *parent)
	: QWidget(parent)
	, m_iWidth(0)
	, m_iHeight(0)
	, m_dpraPtr(nullptr)
	, m_cudpraPtr(nullptr)
{
	ui.setupUi(this);
	ui.computeAIAgroupBox->hide();

	connect(ui.loadphiButton, SIGNAL(clicked()), this, SLOT(openPhi()));
	connect(ui.loadAIAImgButton, SIGNAL(clicked()), this, SLOT(openAIAImages()));
	connect(ui.computeAIAButton, SIGNAL(clicked()), this, SLOT(computeAIA()));
	connect(ui.exitButton, SIGNAL(clicked()), this, SIGNAL(onexit()));
	connect(ui.computeDPRAButton, SIGNAL(clicked()), this, SLOT(computeDPRA()));
	connect(ui.chooseImgButton, SIGNAL(clicked()), this, SLOT(openDPRAImages()));
	connect(ui.outputButton, SIGNAL(clicked()), this, SLOT(outputVideo()));

	ui.outputVideogroupBox->setDisabled(true);
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
	if (!m_AIAImgFileList.isEmpty())
		m_AIAImgFileList.clear();

	// Open AIA images & remember the opened path
	m_AIAImgFileList = QFileDialog::getOpenFileNames(
		this,
		tr("Open AIA images"),
		m_filePath,
		tr("Img (*.png *.bmp *.jpg *.tif *.jpeg)"));

	if (!m_AIAImgFileList.isEmpty())
	{
		QFileInfo fileInfor(m_AIAImgFileList.at(0));
		m_filePath = fileInfor.path();
	}
}

void DPRAWidget::computeAIA()
{
	if (m_AIAImgFileList.isEmpty())
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
	while (!in.atEnd())
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

		if (QMessageBox::Yes == reply)
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

	QMessageBox::information(
		this,
		tr("AIA computation completed"),
		tr("ALA Algorithm on %1 images completed in %2 ms.\n")
		.arg(m_DPRAImgFileList.size()).arg(time) +
		tr("Number of Iterations is: %1\n").arg(iter) + 
		tr("Maximum Error is: %1").arg(err));

	// Enable the DPRA group box
	ui.DPRABox->setEnabled(true);
}

void DPRAWidget::openDPRAImages()
{
	if (!m_DPRAImgFileList.isEmpty())
	{
		m_DPRAImgFileList.clear();
	}

	// Open DPRA images & remember the opened path
	m_DPRAImgFileList = QFileDialog::getOpenFileNames(
		this,
		tr("Open DPRA images"),
		m_filePath,
		tr("Img (*.png *.bmp *.jpg *.tif *.jpeg)"));

	if (!m_DPRAImgFileList.isEmpty())
	{
		QFileInfo fileInfor(m_DPRAImgFileList.at(0));
		m_filePath = fileInfor.path();
	}
}

void DPRAWidget::computeDPRA()
{
	if (m_DPRAImgFileList.isEmpty())
	{
		QMessageBox::critical(this, tr("Error!"), tr("No DPRA images were selected!"));
		return;
	}

	// Get the parameters
	int iupdateRate = ui.rateSpinBox->text().toInt();
	int iNumThreads = ui.threadsSpinBox->text().toInt();

	// Timer
	double dTotaltime = 0;
	double dTime_per_Frame = 0;

	// Final delta phi sum
	if (!m_deltaPhiSum.empty())
	{
		m_deltaPhiSum.clear();
	}
	m_deltaPhiSum.reserve(m_DPRAImgFileList.size());

	
	if (ui.GPUradioButton->isChecked())
	{
		// Construct the DPRA CUDA object
		m_cudpraPtr.reset(new DPRA::DPRA_CUDAF(mv_refPhi.data(), m_iWidth, m_iHeight, iupdateRate));

		// The final deltaPhi
		std::vector<float> deltaPhi(m_iWidth*m_iHeight, 0);

		// Use the progress bar to monitor the progress
		QProgressDialog progress(this);

		progress.setLabelText(tr("DPRA algorithm is running on %1 frames. Please wait...").arg(m_DPRAImgFileList.size()));
		progress.setRange(0, m_DPRAImgFileList.size());
		progress.setModal(true);
		progress.show();

		/* DPRA computation starts */
		int i = 0;
		foreach(const QString& imgstr, m_DPRAImgFileList)
		{
			cv::Mat img = cv::imread(imgstr.toStdString());

			if (img.cols != m_iWidth || img.rows != m_iHeight)
			{
				QMessageBox::critical(this, tr("Error!"), tr("DPRA images must have the same size!"));
				return;
			}

			// Convert the image to grayscale
			cv::cvtColor(img, img, CV_BGRA2GRAY);

			// per-frame computation
			m_cudpraPtr->dpra_per_frame(img, deltaPhi, dTime_per_Frame);

			// Get the time
			dTotaltime += dTime_per_Frame;

			m_deltaPhiSum.push_back(deltaPhi);

			// Update the reference
			if (i % iupdateRate == 0)
				m_cudpraPtr->update_ref_phi();

			progress.setValue(i);
			QApplication::processEvents();

			if (progress.wasCanceled())
			{
				QMessageBox::StandardButton result = QMessageBox::warning(
					this,
					tr("Cancel Processing"),
					tr("Would you like to cancle the execution?\n") +
					tr("Only (%1) of (%2) frames have been computed")
					.arg(i)
					.arg(m_DPRAImgFileList.size()),
					QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
				if (result == QMessageBox::Yes)
				{
					break;
				}
				else
				{
					progress.reset();
				}
			}

			//char str[5];
			//sprintf(str, "%d", i);
			//std::string s = "1_deltaPhi" + std::string(str) + ".bin";

			//WFT_FPA::Utils::WriteMatrixToDisk(s.c_str(), m_iHeight, m_iWidth, m_deltaPhiSum[i].data());


			++i;
		}

		QMessageBox::information(
			this,
			tr("DPRA computation completed"),
			tr("DPRA Algorithm on %1 images of %2 images completed in %3 ms. Average FPS is %4.")
			.arg(i)
			.arg(m_DPRAImgFileList.size())
			.arg(dTotaltime)
			.arg(1000.0 / (dTotaltime / m_DPRAImgFileList.size())));

	}
	else
	{
		// Construct the DPRA hybrid object
		m_dpraPtr.reset(new DPRA::DPRA_HYBRIDF(mv_refPhi.data(), m_iWidth, m_iHeight, iupdateRate, iNumThreads));

		// The final deltaPhi
		std::vector<float> deltaPhi(m_iWidth*m_iHeight, 0);

		// Use the progress bar to monitor the progress
		QProgressDialog progress(this);

		progress.setLabelText(tr("DPRA algorithm is running on %1 frames. Please wait...").arg(m_DPRAImgFileList.size()));
		progress.setRange(0, m_DPRAImgFileList.size());
		progress.setModal(true);
		progress.show();

		/* DPRA computation starts */
		int i = 0;
		foreach(const QString& imgstr, m_DPRAImgFileList)
		{
			cv::Mat img = cv::imread(imgstr.toStdString());

			if (img.cols != m_iWidth || img.rows != m_iHeight)
			{
				QMessageBox::critical(this, tr("Error!"), tr("DPRA images must have the same size!"));
				return;
			}

			// Convert the image to grayscale
			cv::cvtColor(img, img, CV_BGRA2GRAY);

			// per-frame computation
			m_dpraPtr->dpra_per_frame(img, deltaPhi, dTime_per_Frame);

			// Get the time
			dTotaltime += dTime_per_Frame;

			m_deltaPhiSum.push_back(deltaPhi);

			// Update the reference
			if (i % iupdateRate == 0)
				m_dpraPtr->update_ref_phi();

			progress.setValue(i);
			QApplication::processEvents();

			if (progress.wasCanceled())
			{
				QMessageBox::StandardButton result = QMessageBox::warning(
					this,
					tr("Cancel Processing"),
					tr("Would you like to cancle the execution?\n") +
					tr("Only (%1) of (%2) frames have been computed")
					.arg(i)
					.arg(m_DPRAImgFileList.size()),
					QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
				if (result == QMessageBox::Yes)
				{
					break;
				}
				else
				{
					progress.reset();
				}
			}

			//char str[5];
			//sprintf(str, "%d", i);
			//std::string s = "1_deltaPhi" + std::string(str) + ".bin";

			//WFT_FPA::Utils::WriteMatrixToDisk(s.c_str(), m_iHeight, m_iWidth, m_deltaPhiSum[i].data());


			++i;
		}

		QMessageBox::information(
			this,
			tr("DPRA computation completed"),
			tr("DPRA Algorithm on %1 images of %2 images completed in %3 ms. Average FPS is %4.")
			.arg(i)
			.arg(m_DPRAImgFileList.size())
			.arg(dTotaltime)
			.arg(1000.0 / (dTotaltime / m_DPRAImgFileList.size())));
	}


	// Disable the DPRA after computation
	ui.DPRABox->setDisabled(true);
	ui.outputVideogroupBox->setDisabled(false);
}

void DPRAWidget::outputVideo()
{
	// Get the output file name
	QString outputFileName = QFileDialog::getSaveFileName(
		this,
		tr("Save the DPRA results to a video"),
		m_filePath,
		tr("Video (*.mp4 *.wmv *.mpeg *.flv)"));

	if (!outputFileName.isEmpty())
	{
		QFileInfo fileInfor(outputFileName);
		m_filePath = fileInfor.path();


		// Write to video
		if (!videoWritter(outputFileName))
			return;

		ui.outputVideogroupBox->setDisabled(true);

		QMessageBox::StandardButton result = QMessageBox::information(
			this,
			tr("Output Video Completed"),
			tr("The Video is successfully output to ") + outputFileName +
			tr("\nWould you like to play the output video now? "),
			QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);

		if (result == QMessageBox::Yes)
		{
			emit(outputFileNameChanged(outputFileName));
		}
		else
		{
			return;
		}
	}
}

bool DPRAWidget::videoWritter(const QString& fileName)
{
	std::vector<cv::Mat> outFrames;
	outFrames.reserve(m_deltaPhiSum.size());

	QProgressDialog progress(this);

	progress.setLabelText(tr("Writting Video..."));
	progress.setRange(0, m_deltaPhiSum.size());
	progress.setCancelButton(0);
	progress.setModal(true);
	progress.show();

	// Construct video frames
	for (int i = 0; i < m_deltaPhiSum.size(); i++)
	{
		progress.setValue(i);
		QApplication::processEvents();

		std::vector<uchar> v_img;
		v_img.reserve(m_iWidth*m_iHeight);
		for (int j = 0; j < m_deltaPhiSum[0].size(); j++)
		{
			float tempPhi = m_deltaPhiSum[i][j];
			v_img.push_back(uchar((atan2(sin(tempPhi), cos(tempPhi)) + float(M_PI)) / 2.0f / float(M_PI)*255.0f));	
		}

		outFrames.push_back(cv::Mat(m_iHeight, m_iWidth, CV_8UC1, v_img.data()).clone());
	}
	
	// Output the Video
	QFileInfo fileInfor(fileName);
	QString fileExtension = fileInfor.suffix();
	double fps = ui.fpsDoubleSpinBox->text().toDouble();

	if (QString::compare(fileExtension, QLatin1String("mp4")) == 0)
	{
		cv::VideoWriter videoW(
			fileName.toStdString(), 
			CV_FOURCC('D', 'I', 'V', 'X'),
			fps,
			cv::Size(m_iWidth, m_iHeight),
			false);

		for (int i = 0; i < outFrames.size(); i++)
		{
			videoW.write(outFrames[i]);
			cv::imwrite(std::string(m_filePath.toStdString() + "\\" + std::to_string(i) + ".bmp").c_str(), outFrames[i]);
		}
	}
	else if (QString::compare(fileExtension, QLatin1String("flv")) == 0)
	{
		cv::VideoWriter videoW(
			fileName.toStdString(), 
			CV_FOURCC('F', 'L', 'V', '1'), 
			fps,
			cv::Size(m_iWidth, m_iHeight),
			false);

		for (int i = 0; i < outFrames.size(); i++)
		{
			videoW.write(outFrames[i]);
			cv::imwrite(std::string(m_filePath.toStdString() + "\\" + std::to_string(i) + ".bmp").c_str(), outFrames[i]);
		}
	}
	else if (QString::compare(fileExtension, QLatin1String("wmv")) == 0)
	{
		cv::VideoWriter videoW(
			fileName.toStdString(), 
			CV_FOURCC('W', 'M', 'V', '2'), 
			fps,
			cv::Size(m_iWidth, m_iHeight),
			false);

		for (int i = 0; i < outFrames.size(); i++)
		{
			videoW.write(outFrames[i]);
			cv::imwrite(std::string(m_filePath.toStdString() + "\\" + std::to_string(i) + ".bmp").c_str(), outFrames[i]);
		}
	}
	else if (QString::compare(fileExtension, QLatin1String("mpeg")) == 0)
	{
		cv::VideoWriter videoW(
			fileName.toStdString(), 
			CV_FOURCC('P', 'I', 'M', '1'), 
			fps,
			cv::Size(m_iWidth, m_iHeight),
			false);

		for (int i = 0; i < outFrames.size(); i++)
		{
			videoW.write(outFrames[i]);
			cv::imwrite(std::string(m_filePath.toStdString() + "\\" + std::to_string(i) + ".bmp").c_str(), outFrames[i]);
		}
	}
	else
	{
		QMessageBox::critical(
		this,
		tr("Format Error!"),
		tr("Unsupported Video Format!"));

		return false;
	}

	return true;
}