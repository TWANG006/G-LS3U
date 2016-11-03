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

signals:
	void outputFileNameChanged(const QString&);

private slots:
	void openPhi();
	void openAIAImages();
	void openDPRAImages();

private:
	QString m_outputVideoFileName;
	QString m_phiFileName;
	QStringList m_AIAImgFileList;
	QStringList m_DPRAImgFileList;

	Ui::DPRAWidget ui;
};

#endif // DPRAWIDGET_H
