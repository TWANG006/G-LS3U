#include "stdafx.h"
#include "app_dpra.h"


App_DPRA::App_DPRA(QWidget *parent)
	: QMainWindow(parent)
	, m_videoWidget(nullptr)
	, m_playSlider(nullptr)
	, m_playerControl(nullptr)
	, m_dpraWidget(nullptr)
{
	ui.setupUi(this);

	/* Create All needed widgets */
	m_videoWidget = new VideoWidget(this);
	m_playSlider = new QSlider(Qt::Horizontal, this);
	m_playSlider->setRange(0, 100);		// Should be changed later
	m_playerControl = new PlayerControl(this);
	m_dpraWidget = new DPRAWidget(this);

	/* Create Layouts */
	QBoxLayout *videoLayout = new QVBoxLayout;
	videoLayout->addWidget(m_videoWidget,2);
	videoLayout->addWidget(m_playSlider);
	videoLayout->addWidget(m_playerControl);

	QBoxLayout *layout = new QHBoxLayout;
	layout->addLayout(videoLayout);
	layout->addWidget(m_dpraWidget);

	ui.centralWidget->setLayout(layout);


}

App_DPRA::~App_DPRA()
{

}
