#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
X API連携モジュール (tweepy使用版)
OAuth 1.0aを使用したメディアアップロード機能を提供
v1.1 APIでメディアアップロードし、v2 APIでツイート投稿
"""

import os
import time
import json
import webbrowser
from pathlib import Path
import tweepy
import datetime

class TwitterAPITweepy:
    def __init__(self):
        """初期化処理: 認証情報を読み込む"""
        # OAuth 1.0a 認証情報
        self.consumer_key = None
        self.consumer_secret = None
        self.oauth1_access_token = None
        self.oauth1_access_token_secret = None
        
        # 設定を読み込む
        self._load_tokens()
        
        # tweepy クライアント
        self.api = None  # v1.1 API用 (メディアアップロード)
        self.client = None  # v2 API用 (ツイート投稿)
        
        # APIのベースURL
        self.api_base_url = "https://api.twitter.com/2"
    
    def _load_tokens(self):
        """認証情報をファイルから読み込む"""
        try:
            # 設定ファイルパス
            config_path = Path("credentials/twitter_config.json")
            
            # ファイルが存在するかチェック
            if not config_path.exists():
                print(f"設定ファイルが見つかりません: {config_path}")
                return False
                
            # ファイル読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # OAuth 1.0a 設定を読み込み
            self.consumer_key = config.get('consumer_key')
            self.consumer_secret = config.get('consumer_secret')
            self.oauth1_access_token = config.get('oauth1_access_token')
            self.oauth1_access_token_secret = config.get('oauth1_access_token_secret')
            
            return True
            
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            return False
    
    def _save_tokens(self):
        """トークンを設定ファイルに保存する"""
        try:
            # 設定ファイルパス
            config_path = Path("credentials/twitter_config.json")
            
            # 既存の設定を読み込む
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
                
            # 現在のトークンで更新
            config.update({
                'consumer_key': self.consumer_key,
                'consumer_secret': self.consumer_secret,
                'oauth1_access_token': self.oauth1_access_token,
                'oauth1_access_token_secret': self.oauth1_access_token_secret
            })
            
            # ファイルに保存
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
                
            return True
            
        except Exception as e:
            print(f"設定ファイルの保存エラー: {e}")
            return False
    
    def _init_clients(self):
        """API v1.1 と v2 両方のクライアントを初期化"""
        if not self.api or not self.client:
            try:
                # v1.1 API クライアント (メディアアップロード用)
                auth = tweepy.OAuth1UserHandler(
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret,
                    access_token=self.oauth1_access_token,
                    access_token_secret=self.oauth1_access_token_secret
                )
                self.api = tweepy.API(auth)
                
                # v2 API クライアント (ツイート投稿用)
                self.client = tweepy.Client(
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret,
                    access_token=self.oauth1_access_token,
                    access_token_secret=self.oauth1_access_token_secret
                )
                
                return True
            except Exception as e:
                print(f"クライアント初期化エラー: {e}")
                return False
        return True
    
    def _ensure_oauth1_auth(self):
        """OAuth 1.0a認証が設定されているか確認し、設定されていなければ認証フローを開始"""
        if not self.oauth1_access_token or not self.oauth1_access_token_secret:
            print("OAuth 1.0a認証が必要です")
            return self.setup_oauth1()
        
        # クライアントを初期化
        return self._init_clients()
    
    def setup_oauth1(self):
        """OAuth 1.0a認証フローを開始"""
        try:
            # 認証ハンドラを作成
            auth = tweepy.OAuth1UserHandler(
                consumer_key=self.consumer_key,
                consumer_secret=self.consumer_secret,
                callback="http://localhost:8000"
            )
            
            # 認証URLを取得
            auth_url = auth.get_authorization_url()
            
            # URLをコンソールに表示してブラウザで開く
            print("\nOAuth 1.0a認証のためにブラウザでTwitterにログインしてください:")
            print(f"\n{auth_url}\n")
            
            try:
                # ブラウザで認証ページを開く
                webbrowser.open(auth_url)
                print("デフォルトブラウザで認証ページを開きました。")
            except:
                print("ブラウザを自動で開けませんでした。上記URLを手動でブラウザにコピー＆ペーストしてください。")
            
            # コールバックURLの入力を待機
            print("\nTwitterでの認証が完了したら、リダイレクトされたURLをコピーしてください。")
            print("注意: 「このサイトにアクセスできません」というエラー画面が表示されますが、問題ありません。")
            print("ブラウザのアドレスバーに表示されているURL全体をコピーして貼り付けてください。")
            
            callback_url = input("\nリダイレクトされたURL全体を入力してください: ").strip()
            
            # URLから検証コードを抽出
            try:
                # oauth_verifierパラメータをURLから抽出
                import urllib.parse as urlparse
                parsed_url = urlparse.urlparse(callback_url)
                query_params = urlparse.parse_qs(parsed_url.query)
                
                if 'oauth_verifier' in query_params:
                    verifier = query_params['oauth_verifier'][0]
                    print(f"検証コードを取得しました: {verifier}")
                else:
                    print("URLから検証コードを抽出できませんでした。手動で入力してください。")
                    verifier = input("oauth_verifier値を入力: ").strip()
            except:
                print("URLの解析に失敗しました。URLから oauth_verifier パラメータを手動で抽出して入力してください。")
                verifier = input("oauth_verifier値を入力: ").strip()
            
            # アクセストークンを取得
            auth.get_access_token(verifier)
            
            # 認証情報を保存
            self.oauth1_access_token = auth.access_token
            self.oauth1_access_token_secret = auth.access_token_secret
            
            print("\nOAuth 1.0a認証が完了しました！")
            
            # APIクライアントを初期化
            self.api = tweepy.API(auth)
            
            # v2 API クライアントも初期化
            self.client = tweepy.Client(
                consumer_key=self.consumer_key,
                consumer_secret=self.consumer_secret,
                access_token=self.oauth1_access_token,
                access_token_secret=self.oauth1_access_token_secret
            )
            
            # 認証確認
            user = self.api.verify_credentials()
            print(f"認証ユーザー: @{user.screen_name}")
            
            # トークンを保存
            self._save_tokens()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n認証プロセスがユーザーによってキャンセルされました。")
            return False
        except Exception as e:
            print(f"\n認証プロセス中にエラーが発生しました: {e}")
            return False
    
    def check_auth_status(self):
        """認証状態を確認する"""
        # 認証情報の読み込み
        if not self._load_tokens():
            return False
            
        # OAuth 1.0a の認証確認
        if self.consumer_key and self.consumer_secret and self.oauth1_access_token and self.oauth1_access_token_secret:
            try:
                auth = tweepy.OAuth1UserHandler(
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret,
                    access_token=self.oauth1_access_token,
                    access_token_secret=self.oauth1_access_token_secret
                )
                self.api = tweepy.API(auth)
                user = self.api.verify_credentials()
                print(f"OAuth 1.0a認証: 有効 (@{user.screen_name})")
                
                # v2 API クライアントも初期化
                self.client = tweepy.Client(
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret,
                    access_token=self.oauth1_access_token,
                    access_token_secret=self.oauth1_access_token_secret
                )
                
                return True
            except Exception as e:
                print(f"OAuth 1.0a認証: 無効 ({e})")
                return False
        else:
            print("OAuth 1.0a認証: 認証情報が不足しています")
            return False
    
    def post_tweet(self, text, image_paths=None):
        """
        ツイートを投稿する (v1.1 API でメディアアップロード + v2 API でツイート)
        
        Args:
            text: ツイート本文
            image_paths: 添付する画像のパスのリスト（オプション）
            
        Returns:
            str or None: 投稿に成功した場合はツイートID、失敗した場合はNone
        """
        # OAuth 1.0a 認証確認
        if not self._ensure_oauth1_auth():
            print("OAuth 1.0a認証が必要です。python main.py --auth_oauth1 を実行してください")
            return None
            
        try:
            # 画像がある場合はアップロード (v1.1 API)
            media_ids = []
            if image_paths:
                # リストでない場合はリストに変換（後方互換性のため）
                if isinstance(image_paths, (str, Path)):
                    image_paths = [image_paths]
                    
                for image_path in image_paths:
                    try:
                        # 画像パスが相対パスの場合は絶対パスに変換
                        if not os.path.isabs(image_path):
                            image_path = os.path.abspath(image_path)
                        
                        # ファイルを確認
                        image_path = Path(image_path)
                        if not image_path.exists():
                            print(f"画像ファイルが見つかりません: {image_path}")
                            continue
                        
                        print(f"画像をアップロード中: {image_path}")
                        media = self.api.media_upload(filename=str(image_path))
                        media_ids.append(str(media.media_id))
                        print(f"画像がアップロードされました。メディアID: {media.media_id}")
                        
                    except Exception as e:
                        print(f"画像のアップロードに失敗しました: {image_path} - {str(e)}")
                        continue
            
            # ツイートを投稿 (v2 API)
            print(f"ツイート内容: {text}")
            if media_ids:
                print(f"添付メディア: {media_ids}")
                response = self.client.create_tweet(
                    text=text,
                    media_ids=media_ids
                )
            else:
                response = self.client.create_tweet(
                    text=text
                )
            
            tweet_id = response.data['id']
            print(f"ツイートが投稿されました！ ID: {tweet_id}")
            print(f"https://twitter.com/user/status/{tweet_id}")
            
            return tweet_id
            
        except Exception as e:
            print(f"ツイート投稿エラー: {e}")
            return None
    
    def _extract_rate_limit_info(self, response):
        """
        レスポンスヘッダーからレート制限情報を抽出する
        
        Args:
            response: requestsのレスポンスオブジェクト
            
        Returns:
            dict: レート制限情報を含む辞書
        """
        headers = response.headers
        rate_limit_info = {
            'reset': headers.get('x-rate-limit-reset'),
            'remaining': headers.get('x-rate-limit-remaining'),
            'limit': headers.get('x-rate-limit-limit')
        }
        
        # エポックタイムをフォーマット
        if rate_limit_info['reset']:
            epoch_time = int(rate_limit_info['reset'])
            rate_limit_info['formatted_time'] = self._format_epoch_time(epoch_time)
        
        return rate_limit_info
    
    def _format_epoch_time(self, epoch_time):
        """
        エポックタイムを読みやすい形式に変換する
        
        Args:
            epoch_time: UNIXエポックタイム
            
        Returns:
            dict: フォーマットされた時間情報
        """
        # 現在時刻
        now = datetime.datetime.now()
        
        # エポック時刻をdatetimeに変換
        reset_time = datetime.datetime.fromtimestamp(epoch_time)
        
        # 日本時間表示
        japan_time = reset_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 待機時間計算
        wait_seconds = max(0, (reset_time - now).total_seconds())
        minutes, seconds = divmod(int(wait_seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            wait_str = f"あと{hours}時間{minutes}分{seconds}秒"
        elif minutes > 0:
            wait_str = f"あと{minutes}分{seconds}秒"
        else:
            wait_str = f"あと{seconds}秒"
        
        return {
            'epoch_time': epoch_time,
            'japan_time': japan_time,
            'wait_time': wait_str
        }
    
    def _print_rate_limit_info(self, rate_limit_info):
        """
        レート制限情報を表示する
        
        Args:
            rate_limit_info: レート制限情報を含む辞書
        """
        if not rate_limit_info.get('reset'):
            # ヘッダーに情報がない場合は何も表示しない
            return
        
        print("\n=== Twitter API レート制限情報 ===")
        print(f"利用可能リクエスト数: {rate_limit_info.get('remaining', 'N/A')}/{rate_limit_info.get('limit', 'N/A')}")
        
        if 'formatted_time' in rate_limit_info:
            time_info = rate_limit_info['formatted_time']
            print(f"リセット時刻: {time_info['japan_time']} ({time_info['wait_time']})")
            print(f"エポックタイム: {time_info['epoch_time']}")
        
        print("=============================\n") 