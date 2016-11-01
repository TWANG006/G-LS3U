#ifndef APP_DPRA_H
#define APP_DPRA_H

#include <QtWidgets/QMainWindow>
#include "ui_app_dpra.h"

class App_DPRA : public QMainWindow
{
	Q_OBJECT

public:
	App_DPRA(QWidget *parent = 0);
	~App_DPRA();

private:
	Ui::App_DPRAClass ui;
};

#endif // APP_DPRA_H
