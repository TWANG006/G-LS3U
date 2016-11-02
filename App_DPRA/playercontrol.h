#ifndef PLAYERCONTROL_H
#define PLAYERCONTROL_H

#include <QWidget>
#include "ui_playercontrol.h"

class PlayerControl : public QWidget
{
	Q_OBJECT

public:
	PlayerControl(QWidget *parent = 0);
	~PlayerControl();

private:
	Ui::PlayerControl ui;
};

#endif // PLAYERCONTROL_H
