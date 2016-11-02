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

private:
	VideoWidget *m_videoWidget;
	PlayerControl *m_playerControl;
	DPRAWidget *m_dpraWidget;

	QSlider *m_playSlider;


	Ui::App_DPRAClass ui;
};

#endif // APP_DPRA_H
