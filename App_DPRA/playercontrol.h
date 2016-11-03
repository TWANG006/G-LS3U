#ifndef PLAYERCONTROL_H
#define PLAYERCONTROL_H

#include <QWidget>
#include "ui_playercontrol.h"

#include <QMediaPlayer>

class PlayerControl : public QWidget
{
	Q_OBJECT

public:
	PlayerControl(QWidget *parent = 0);
	~PlayerControl();

	QMediaPlayer::State state() const;
	int volume() const;
	bool isMuted() const;
	qreal playbackRate() const;

public slots:
	void setState(QMediaPlayer::State state);
	void setVolume(int volume);
	void setMuted(bool muted);
	void setFullScreenButtonChecked(bool checked);

signals:
	void play();
	void pause();
	void stop();
	void fullScreen(bool isFullScreen);
	void videoOpen();
	void changeVolume(int volume);
	void changeMuting(bool muting);
	void changeRate(qreal rate);

private slots:
	void playClicked();
	void muteClicked();
	void updateRate();

private:
	Ui::PlayerControl ui;

	int m_volume;
	bool m_playerMuted;
	QMediaPlayer::State m_playerState;
};

#endif // PLAYERCONTROL_H
