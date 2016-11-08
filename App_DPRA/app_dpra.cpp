#include "stdafx.h"
#include "app_dpra.h"


App_DPRA::App_DPRA(QWidget *parent)
	: QMainWindow(parent)
	, m_player(nullptr)
	, m_videoWidget(nullptr)
	, m_playSlider(nullptr)
	, m_playerControl(nullptr)
	, m_dpraWidget(nullptr)
{
	ui.setupUi(this);

	/* Create the media player */
	m_player = new QMediaPlayer(this);

	/* Create All needed widgets */
	m_videoWidget = new VideoWidget(this);
	m_player->setVideoOutput(m_videoWidget);	
	
	m_playerControl = new PlayerControl(this);
	m_playerControl->setState(m_player->state());
	m_playerControl->setVolume(m_player->volume());
	m_playerControl->setMuted(m_playerControl->isMuted());

	m_dpraWidget = new DPRAWidget(this);

	m_playSlider = new QSlider(Qt::Horizontal, this);
	m_durationLabel = new QLabel(this);	
	m_playSlider->setRange(0, m_player->duration() / 1000);


	/* Create Layouts */
	QBoxLayout *hLayout = new QHBoxLayout;
	hLayout->addWidget(m_playSlider);
	hLayout->addWidget(m_durationLabel);

	QBoxLayout *videoLayout = new QVBoxLayout;
	videoLayout->addWidget(m_videoWidget,2);
	videoLayout->addLayout(hLayout);
	videoLayout->addWidget(m_playerControl);

	QBoxLayout *layout = new QHBoxLayout;
	layout->addWidget(m_dpraWidget);
	layout->addLayout(videoLayout, 2);
	ui.centralWidget->setLayout(layout);

	/* Connect signal/slot */
	connect(m_player, &QMediaPlayer::durationChanged, this, &App_DPRA::durationChanged);
	connect(m_player, &QMediaPlayer::positionChanged, this, &App_DPRA::positionChanged);
	connect(m_player, SIGNAL(metaDataChanged()), SLOT(metaDataChanged()));
	connect(m_player, &QMediaPlayer::videoAvailableChanged, this, &App_DPRA::videoAvailableChanged);

	connect(m_playSlider, &QSlider::sliderMoved, this, &App_DPRA::seek);

	connect(m_playerControl, &PlayerControl::videoOpen, this, &App_DPRA::openVideo);
	connect(m_playerControl, &PlayerControl::play, m_player, &QMediaPlayer::play);
	connect(m_playerControl, &PlayerControl::pause, m_player, &QMediaPlayer::pause);
	connect(m_playerControl, &PlayerControl::stop, m_player, &QMediaPlayer::stop);
	connect(m_playerControl, SIGNAL(stop()), m_videoWidget, SLOT(update()));
	connect(m_playerControl, &PlayerControl::changeVolume, m_player, &QMediaPlayer::setVolume);
	connect(m_playerControl, &PlayerControl::changeRate, m_player, &QMediaPlayer::setPlaybackRate);

	connect(m_player, SIGNAL(stateChanged(QMediaPlayer::State)), m_playerControl, SLOT(setState(QMediaPlayer::State)));
	connect(m_player, SIGNAL(volumeChanged(int)), m_playerControl, SLOT(setVolume(int)));
	connect(m_player, &QMediaPlayer::stateChanged, m_playerControl, &PlayerControl::setState);
	connect(m_playerControl, &PlayerControl::changeMuting, m_player, &QMediaPlayer::setMuted);
	connect(m_player, &QMediaPlayer::mutedChanged, m_playerControl, &PlayerControl::setMuted);
	connect(m_player, &QMediaPlayer::mediaStatusChanged, this, &App_DPRA::statusChanged);

	connect(m_dpraWidget, &DPRAWidget::outputFileNameChanged, this, &App_DPRA::playVideoFile);

	connect(m_dpraWidget, &DPRAWidget::onexit, this, &App_DPRA::close);

	metaDataChanged();
}

App_DPRA::~App_DPRA()
{
	m_player->stop();
}

void App_DPRA::playVideoFile(const QString& qstr)
{
	m_videoFileName = qstr;

	QFileInfo fileInfo(m_videoFileName);
	m_filePath = fileInfo.path();
	m_player->setMedia(QUrl::fromLocalFile(fileInfo.absoluteFilePath()));
	m_player->play();
}

void App_DPRA::openVideo()
{
	// Open video & remember the last opened path
	m_videoFileName = QFileDialog::getOpenFileName(this, tr("Open Video Files"), m_filePath, tr("Videos (*.mp4 *.wmv *.mpeg *.flv)"));
	if (!m_videoFileName.isNull())
	{
		QFileInfo fileInfo(m_videoFileName);
		m_filePath = fileInfo.path();
		m_player->setMedia(QUrl::fromLocalFile(fileInfo.absoluteFilePath()));
		m_player->play();
	}
}

void App_DPRA::durationChanged(qint64 duration)
{
	m_duration = duration / 1000;
	m_playSlider->setMaximum(duration / 1000);
}
void App_DPRA::positionChanged(qint64 progress)
{
	if (!m_playSlider->isSliderDown()) {
        m_playSlider->setValue(progress / 1000);
    }
    updateDurationInfo(progress / 1000);
}
void App_DPRA::metaDataChanged()
{
	if (m_player->isMetaDataAvailable())
	{
		setTrackInfo(QString("%1 - %2")
			.arg(m_player->metaData(QMediaMetaData::AlbumArtist).toString())
			.arg(m_player->metaData(QMediaMetaData::Title).toString()));
	}
}

void App_DPRA::seek(int seconds)
{
	m_player->setPosition(seconds * 1000);
}

void App_DPRA::statusChanged(QMediaPlayer::MediaStatus status)
{
	handleCursor(status);

	// handle status message
    switch (status) {
    case QMediaPlayer::UnknownMediaStatus:
    case QMediaPlayer::NoMedia:
    case QMediaPlayer::LoadedMedia:
    case QMediaPlayer::BufferingMedia:
    case QMediaPlayer::BufferedMedia:
		setStatusInfo(QString());
        break;
    case QMediaPlayer::LoadingMedia:
		setStatusInfo(tr("Loading..."));
        break;
    case QMediaPlayer::StalledMedia:
        setStatusInfo(tr("Media Stalled"));
        break;
    case QMediaPlayer::EndOfMedia:
        QApplication::alert(this);
        break;
    case QMediaPlayer::InvalidMedia:
		displayErrorMessage();
        break;
    }
}

void App_DPRA::displayErrorMessage()
{
	setStatusInfo(m_player->errorString());
}

void App_DPRA::videoAvailableChanged(bool available)
{
	if (!available)
	{
		disconnect(m_playerControl, &PlayerControl::fullScreen, m_videoWidget, &VideoWidget::setFullScreen);
		disconnect(m_videoWidget, &VideoWidget::fullScreenChanged, m_playerControl, &PlayerControl::setFullScreenButtonChecked);
		m_videoWidget->setFullScreen(false);
	}
	else
	{
		connect(m_playerControl, &PlayerControl::fullScreen, m_videoWidget, &VideoWidget::setFullScreen);
		connect(m_videoWidget, &VideoWidget::fullScreenChanged, m_playerControl, &PlayerControl::setFullScreenButtonChecked);
	}
}

void App_DPRA::changeVideoFileName(const QString& qstr)
{
	m_videoFileName = qstr;

	if (!m_videoFileName.isNull())
	{
		QFileInfo fileInfo(m_videoFileName);
		m_filePath = fileInfo.path();
		m_player->setMedia(QUrl::fromLocalFile(fileInfo.absoluteFilePath()));
		m_player->play();
	}
}

void App_DPRA::setTrackInfo(const QString &infor)
{
	m_trackInfo = infor;

	if (!m_statusInfo.isEmpty())
	{
		setWindowTitle(QString("%1 | %2").arg(m_trackInfo).arg(m_statusInfo));
	}
	else
	{
		setWindowTitle(m_trackInfo);
	}
}

void App_DPRA::setStatusInfo(const QString &infor)
{
	m_statusInfo = infor;
	if (!m_statusInfo.isEmpty())
		setWindowTitle(QString("%1 | %2").arg(m_trackInfo).arg(m_statusInfo));
	else
		setWindowTitle(m_trackInfo);
}

void App_DPRA::updateDurationInfo(qint64 currentInfo)
{
	QString tStr;

	if (currentInfo || m_duration)
	{
		QTime currentTime((currentInfo / 3600) % 60, 
			(currentInfo / 60) % 60, 
			currentInfo % 60, 
			(currentInfo * 1000) % 1000);
		QTime totalTime((m_duration / 3600) % 60, 
			(m_duration / 60) % 60, 
			m_duration % 60,
			(m_duration / 1000) % 1000);

		QString format = "mm:ss";

		if (m_duration > 3600)
			format = "hh:mm:ss";

		tStr = currentTime.toString(format) + " / " + totalTime.toString(format);
	}

	m_durationLabel->setText(tStr);
}

void App_DPRA::handleCursor(QMediaPlayer::MediaStatus status)
{
	if (status == QMediaPlayer::LoadingMedia ||
		status == QMediaPlayer::BufferingMedia ||
		status == QMediaPlayer::StalledMedia)
	{
		setCursor(QCursor(Qt::BusyCursor));
	}
	else
	{
		unsetCursor();
	}
}