#include "stdafx.h"
#include "playercontrol.h"

PlayerControl::PlayerControl(QWidget *parent)
	: QWidget(parent)
	, m_playerState(QMediaPlayer::StoppedState)
	, m_playerMuted(false)
{
	ui.setupUi(this);

	ui.rateBox->addItem("0.5x", QVariant(0.5));
    ui.rateBox->addItem("1.0x", QVariant(1.0));
    ui.rateBox->addItem("2.0x", QVariant(2.0));
    ui.rateBox->setCurrentIndex(1);

	// Do the signal/slot connections
	connect(ui.playButton, SIGNAL(clicked()), this, SLOT(playClicked()));
	connect(ui.stopButton, SIGNAL(clicked()), this, SIGNAL(stop()));
	connect(ui.muteButton, SIGNAL(clicked()), this, SLOT(muteClicked()));
	connect(ui.volumeSlider, SIGNAL(sliderMoved(int)), this, SIGNAL(changeVolume(int)));
	connect(ui.rateBox, SIGNAL(activated(int)), this, SLOT(updateRate()));
	connect(ui.fullscreenButton, SIGNAL(clicked(bool)), this, SIGNAL(fullScreen(bool)));
	connect(ui.openButton, SIGNAL(clicked()), this, SIGNAL(videoOpen()));
}

QMediaPlayer::State PlayerControl::state() const
{
	return m_playerState;
}

void PlayerControl::setState(QMediaPlayer::State state)
{
	if (state != m_playerState)
	{
		m_playerState = state;

		switch (state)
		{
		case QMediaPlayer::StoppedState:
			ui.stopButton->setEnabled(false);
			ui.playButton->setIcon(QIcon(QStringLiteral(":/App_DPRA/icon-play")));
			break;
		case QMediaPlayer::PlayingState:
			ui.stopButton->setEnabled(true);
			ui.playButton->setIcon(QIcon(QStringLiteral(":/App_DPRA/icon-pause")));
			break;
		case QMediaPlayer::PausedState:
			ui.stopButton->setEnabled(true);
			ui.playButton->setIcon(QIcon(QStringLiteral(":/App_DPRA/icon-play")));
			break;
		default:
			break;
		}
	}
}

int PlayerControl::volume() const
{
	return ui.volumeSlider ? ui.volumeSlider->value() : 0;
}

void PlayerControl::setVolume(int vol)
{
	if (ui.volumeSlider)
	{
		ui.volumeSlider->setValue(vol);
	}
}

bool PlayerControl::isMuted() const
{
	return m_playerMuted;
}

void PlayerControl::setMuted(bool muted)
{
	if (muted != m_playerMuted)
	{
		m_playerMuted = muted;

		ui.muteButton->setIcon(QIcon(m_playerMuted 
			? QStringLiteral(":/App_DPRA/icon-volmute") 
			: QStringLiteral(":/App_DPRA/icon-vol")));
	}
}

void PlayerControl::playClicked()
{
	switch (m_playerState)
	{
	case QMediaPlayer::StoppedState:
	case QMediaPlayer::PausedState:
		emit play();
		break;
	case QMediaPlayer::PlayingState:
		emit pause();
		break;
	default:
		break;
	}
}

void PlayerControl::muteClicked()
{
	emit changeMuting(!m_playerMuted);
}

qreal PlayerControl::playbackRate() const
{
	return ui.rateBox->itemData(ui.rateBox->currentIndex()).toDouble();
}

void PlayerControl::updateRate()
{
	emit changeRate(playbackRate());
}

void PlayerControl::setFullScreenButtonChecked(bool checked)
{
	ui.fullscreenButton->setChecked(checked);
}

PlayerControl::~PlayerControl()
{

}
