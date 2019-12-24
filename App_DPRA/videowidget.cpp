#include "stdafx.h"
#include "videowidget.h"

VideoWidget::VideoWidget(QWidget *parent)
	: QVideoWidget(parent)
{
	setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	setMinimumWidth(640);
	setMinimumHeight(480);

	QPalette p = palette();
	p.setColor(QPalette::Window, Qt::black);
	
	setPalette(p);

	setAttribute(Qt::WA_OpaquePaintEvent);
}

VideoWidget::~VideoWidget()
{

}

void VideoWidget::keyPressEvent(QKeyEvent *et)
{
	if(et->key() == Qt::Key_Escape && isFullScreen())
	{
		setFullScreen(false);
		et->accept();
	}
	else if(et->key() == Qt::Key_Enter && et->modifiers() & Qt::Key_Alt)
	{
		setFullScreen(!isFullScreen());
		et->accept();
	}
	else
	{
		QVideoWidget::keyPressEvent(et);
	}
}

void VideoWidget::mouseDoubleClickEvent(QMouseEvent *et)
{
	setFullScreen(!isFullScreen());
	et->accept();
}

void VideoWidget::mousePressEvent(QMouseEvent *et)
{
	QVideoWidget::mousePressEvent(et);
}