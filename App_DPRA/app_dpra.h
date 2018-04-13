#ifndef APP_DPRA_H
#define APP_DPRA_H

#include <QtWidgets/QMainWindow>
#include "ui_app_dpra.h"

#include "videowidget.h"
#include "playercontrol.h"
#include "dprawidget.h"

#include <QMediaPlayer>


class App_DPRA : public QMainWindow
{
	Q_OBJECT

public:
	App_DPRA(QWidget *parent = 0);
	~App_DPRA();

protected:
	void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;

public slots:
	void playVideoFile(const QString& qstr);

private slots:
	void openVideo();
	void durationChanged(qint64 duration);
	void positionChanged(qint64 progress);
	void metaDataChanged();

	void seek(int seconds);
	void statusChanged(QMediaPlayer::MediaStatus status);
	void displayErrorMessage();
	void videoAvailableChanged(bool available);

	void changeVideoFileName(const QString& qstr);
private:
	void setTrackInfo(const QString &infor);
	void updateDurationInfo(qint64 currentInfo);
	void setStatusInfo(const QString &infor);
	void handleCursor(QMediaPlayer::MediaStatus status);

private:
	/* GUI related parameters */
	QString m_videoFileName;
	QString m_filePath;

	QMediaPlayer *m_player;
	VideoWidget *m_videoWidget;
	PlayerControl *m_playerControl;
	DPRAWidget *m_dpraWidget;

	QSlider *m_playSlider;
	QLabel *m_durationLabel;

	QString m_trackInfo;
	QString m_statusInfo;
	qint64 m_duration;
	
	Ui::App_DPRAClass ui;
};

#endif // APP_DPRA_H
