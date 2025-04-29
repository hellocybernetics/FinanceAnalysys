#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Twitterハンドラーモジュール - メインアプリケーションのTwitter関連機能を処理
OAuth 1.0a 認証を使用して Twitter API v1.1 (メディア) と v2 (ツイート) にアクセス
"""

import os

def handle_oauth1_auth(twitter_api_tweepy):
    """Twitter OAuth 1.0a認証を処理"""
    print("Twitter OAuth 1.0a認証を開始します...")
    
    try:
        if twitter_api_tweepy.setup_oauth1():
            print("OAuth 1.0a認証が完了しました！")
            print("\n次のようにツイートを投稿できます:")
            print("  python main.py --tweet \"こんにちは、世界！\"")
            print("\n画像付きツイートの例:")
            print("  python main.py --tweet \"素敵な画像です！\" --image \"path/to/image.jpg\"")
            return True
        else:
            print("\n認証に失敗しました。以下をご確認ください:")
            print("1. インターネット接続が安定していること")
            print("2. Twitter開発者ポータルの設定が正しいこと:")
            print("   - コールバックURLが http://localhost:8000 に設定されていること")
            print("   - アプリにRead/Write権限が付与されていること")
            print("3. credentials/twitter_config.jsonのAPIキー情報が正しいこと")
            print("\nもう一度認証を試すには:\n  python main.py --auth\n")
            return False
    except KeyboardInterrupt:
        print("\n\n認証プロセスがユーザーによってキャンセルされました。")
        print("後で認証するには次のコマンドを実行してください:\n  python main.py --auth\n")
        return False
    except Exception as e:
        print(f"\n認証プロセス中にエラーが発生しました: {e}")
        print("問題が解決しない場合は、Twitter開発者ポータルで認証情報を再確認してください。")
        print("コールバックURLが http://localhost:8000 に設定されていることを確認してください。")
        print("もう一度認証を試すには:\n  python main.py --auth\n")
        return False


def handle_tweet_tweepy(args, twitter_api_tweepy):
    """
    テキスト（および画像）をツイートとして投稿
    
    Args:
        args: コマンドライン引数
        twitter_api_tweepy: TwitterAPITweepyインスタンス
    """
    # 認証状態をチェック
    if not twitter_api_tweepy.check_auth_status():
        print("Twitter OAuth 1.0a認証が必要です。以下のコマンドを実行してください。")
        print("python main.py --auth")
        return
        
    tweet_text = args.tweet
    image_paths = args.image.split(',') if args.image else None
    
    # 画像パスが相対パスの場合は絶対パスに変換
    if image_paths:
        image_paths = [os.path.abspath(path.strip()) if not os.path.isabs(path.strip()) else path.strip() 
                      for path in image_paths]
    
    print(f"ツイート内容: {tweet_text}")
    if image_paths:
        print(f"添付画像: {', '.join(image_paths)}")
        
    # 最終確認（forceがTrueの場合はスキップ）
    if not hasattr(args, 'force') or not args.force:
        confirm = input("このツイートを投稿しますか？ (y/n): ").strip().lower()
        if confirm != 'y':
            print("ツイートを中止しました。")
            return
    else:
        print("確認をスキップしてツイートを投稿します...")
    
    # Twitterに投稿
    tweet_id = twitter_api_tweepy.post_tweet(tweet_text, image_paths)
    
    if tweet_id:
        print(f"ツイートを投稿しました！ ID: {tweet_id}")
        print(f"https://twitter.com/user/status/{tweet_id}")
    else:
        print("ツイートの投稿に失敗しました。")