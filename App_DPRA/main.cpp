#include "stdafx.h"
#include "app_dpra.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	qRegisterMetaType<QStringList>("QStringList");
	qRegisterMetaType< std::vector<float> >("std::vector<float>");

	QApplication a(argc, argv);
	App_DPRA w;
	w.show();
	return a.exec();
}
