#ifndef VIDEOWIDGET_H
#define VIDEOWIDGET_H

#include <QVideoWidget>

class VideoWidget : public QVideoWidget
{
	Q_OBJECT

public:
	VideoWidget(QWidget *parent);
	~VideoWidget();

protected:
	void keyPressEvent(QKeyEvent *et);
	void mouseDoubleClickEvent(QMouseEvent *et);
	void mousePressEvent(QMouseEvent *et);

private:
	
};

#endif // VIDEOWIDGET_H
