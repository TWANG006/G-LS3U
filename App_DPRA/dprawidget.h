#ifndef DPRAWIDGET_H
#define DPRAWIDGET_H

#include <QWidget>
#include "ui_dprawidget.h"

class DPRAWidget : public QWidget
{
	Q_OBJECT

public:
	DPRAWidget(QWidget *parent = 0);
	~DPRAWidget();

private:
	Ui::DPRAWidget ui;
};

#endif // DPRAWIDGET_H
